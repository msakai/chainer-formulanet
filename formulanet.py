import chainer
from chainer import dataset
import h5py
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import holstep
import parser_funcparselib
import tree

from torch import nn
import torch.nn.functional as F
import torch


Array = np.ndarray
VariableOrArray = Union[torch.Tensor, Array]


DIM = 256

    
class GraphData(NamedTuple):
    labels: Array
    edges: Array
    treelets: Array


class GraphsData(NamedTuple):
    node_ranges: Array
    labels: Array
    edges: Array
    treelets: Array
    MI: torch.sparse.FloatTensor
    MO: torch.sparse.FloatTensor
    ML: torch.sparse.FloatTensor
    MH: torch.sparse.FloatTensor
    MR: torch.sparse.FloatTensor


class FP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DIM, DIM)
        self.bn = nn.BatchNorm1d(DIM)

    def forward(self, x: VariableOrArray) -> torch.Tensor:
        return F.relu(self.bn(self.fc(x)))


class Block(nn.Module):
    def __init__(self, n_input: int) -> None:
        super().__init__()
        self._n_input = n_input
        self.fc1 = nn.Linear(DIM * n_input, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.bn1 = nn.BatchNorm1d(DIM)
        self.bn2 = nn.BatchNorm1d(DIM)

    def forward(self, *args: VariableOrArray) -> torch.Tensor:
        assert len(args) == self._n_input
        h = F.relu(self.bn1(self.fc1(torch.cat(args, dim=1))))
        h = F.relu(self.bn2(self.fc2(h)))
        return h


class Block2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1a = nn.Linear(DIM, DIM, bias=False)
        self.fc1b = nn.Linear(DIM, DIM, bias=True)
        self.fc2 = nn.Linear(DIM, DIM)
        self.bn1 = nn.BatchNorm1d(DIM)
        self.bn2 = nn.BatchNorm1d(DIM)

    def forward(self, arg: VariableOrArray) -> torch.Tensor:
        h = F.relu(self.bn1(arg))
        h = F.relu(self.bn2(self.fc2(h)))
        return h


class Block3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1a = nn.Linear(DIM, DIM, bias=False)
        self.fc1b = nn.Linear(DIM, DIM, bias=False)
        self.fc1c = nn.Linear(DIM, DIM, bias=True)
        self.fc2 = nn.Linear(DIM, DIM)
        self.bn1 = nn.BatchNorm1d(DIM)
        self.bn2 = nn.BatchNorm1d(DIM)

    def forward(self, arg: VariableOrArray) -> torch.Tensor:
        h = F.relu(self.bn1(arg))
        h = F.relu(self.bn2(self.fc2(h)))
        return h


class Step(nn.Module):
    def __init__(self, order_preserving: bool) -> None:
        super().__init__()
        self._order_preserving = order_preserving
        self.FP = FP()
        self.FI = Block2()
        self.FO = Block2()
        if order_preserving:
            self.FH = Block3()
            self.FL = Block3()
            self.FR = Block3()

    def forward(self, gs: GraphsData, x: VariableOrArray) -> torch.Tensor:
        x_new = x

        FI_fc1a_x = self.FI.fc1a(x)
        FI_fc1b_x = self.FI.fc1b(x)
        FO_fc1a_x = self.FO.fc1a(x)
        FO_fc1b_x = self.FO.fc1b(x)
        FI_inputs = FI_fc1a_x[gs.edges[:, 0]] + FI_fc1b_x[gs.edges[:, 1]]
        FO_inputs = FO_fc1a_x[gs.edges[:, 0]] + FO_fc1b_x[gs.edges[:, 1]]

        FI_outputs = self.FI(FI_inputs)
        FO_outputs = self.FO(FO_inputs)

        d = torch.sparse.mm(gs.MI, FI_outputs) + \
            torch.sparse.mm(gs.MO, FO_outputs)

        x_new = x_new + d

        if self._order_preserving:
            FL_fc1a_x = self.FL.fc1a(x)
            FL_fc1b_x = self.FL.fc1b(x)
            FL_fc1c_x = self.FL.fc1c(x)
            FH_fc1a_x = self.FH.fc1a(x)
            FH_fc1b_x = self.FH.fc1b(x)
            FH_fc1c_x = self.FH.fc1c(x)
            FR_fc1a_x = self.FR.fc1a(x)
            FR_fc1b_x = self.FR.fc1b(x)
            FR_fc1c_x = self.FR.fc1c(x)
            FL_inputs = FL_fc1a_x[gs.treelets[:, 0]] + FL_fc1b_x[gs.treelets[:, 1]] + FL_fc1c_x[gs.treelets[:, 2]]
            FH_inputs = FH_fc1a_x[gs.treelets[:, 0]] + FH_fc1b_x[gs.treelets[:, 1]] + FH_fc1c_x[gs.treelets[:, 2]]
            FR_inputs = FR_fc1a_x[gs.treelets[:, 0]] + FR_fc1b_x[gs.treelets[:, 1]] + FR_fc1c_x[gs.treelets[:, 2]]

            FL_outputs = self.FL(FL_inputs)
            FH_outputs = self.FH(FH_inputs)
            FR_outputs = self.FR(FR_inputs)

            d = torch.sparse.mm(gs.ML, FL_outputs.unsqueeze(1)) + \
                torch.sparse.mm(gs.MH, FH_outputs.unsqueeze(1)) + \
                torch.sparse.mm(gs.MR, FR_outputs.unsqueeze(1))
            d = d.squeeze(1)

            x_new = x_new + d

        return self.FP(x_new)


class Classifier(nn.Module):
    def __init__(self, conditional: bool = True) -> None:
        super().__init__()
        self._conditional = conditional
        self.fc1 = nn.Linear(2 * DIM if conditional else DIM, DIM)
        self.bn = nn.BatchNorm1d(DIM)
        self.fc2 = nn.Linear(DIM, 2)

    def forward(self, *args: VariableOrArray) -> torch.Tensor:
        if self._conditional:
            assert len(args) == 2
        else:
            assert len(args) == 1
        return self.fc2(F.relu(self.bn(self.fc1(torch.cat(args, dim=1)))))


class FormulaNet(nn.Module):
    def __init__(self, vocab_size: int, steps: int, order_preserving: bool, conditional: bool) -> None:
        super().__init__()
        self._order_preserving = order_preserving
        self._conditional = conditional
        self.embed_id = nn.Embedding(vocab_size, DIM)
        self.steps = nn.ModuleList([Step(order_preserving) for _ in range(steps)])
        self.classifier = Classifier(conditional)

    def forward(self, tmp) -> torch.Tensor:
        # tmp: Tuple[GraphsData, List[Tuple[int, int, bool]]], labels: Array
        gs, minibatch = tmp
        stmt_embeddings = []
        conj_embeddings = []

        def collect_embedding() -> None:
            es = [self._compute_graph_embedding(gs, x, j) for j in range(len(gs.node_ranges))]
            for (conj, stmt, y) in minibatch:
                stmt_embeddings.append(es[stmt])
                if self._conditional:
                    conj_embeddings.append(es[conj])

        x = self._initial_nodes_embedding(gs)
        collect_embedding()
        for (i, step) in enumerate(self.steps):
            x = step(gs, x)
            collect_embedding()

        if self._conditional:
            predicted = self.classifier(torch.cat(conj_embeddings, dim=0), torch.cat(stmt_embeddings, dim=0))
        else:
            predicted = self.classifier(torch.cat(stmt_embeddings, dim=0))

        return predicted

    def predict(self, gs: GraphsData, conj: int, stmt: int) -> bool:
        return F.argmax(self.logit(gs, conj, stmt)) > 0

    def logit(self, gs: GraphsData, conj: int, stmt: int) -> torch.Tensor:
        x = self._initial_nodes_embedding(gs)
        for (i, step) in enumerate(self.steps):
            x = step(gs, x)

        stmt_embedding = self._compute_graph_embedding(gs, x, stmt)
        if self._conditional:
            conj_embedding = self._compute_graph_embedding(gs, x, conj)
            return self.classifier(conj_embedding, stmt_embedding)[0]
        else:
            return self.classifier(stmt_embedding)[0]

    def _initial_nodes_embedding(self, gs: GraphsData) -> torch.Tensor:
        return self.embed_id(gs.labels)

    def _compute_graph_embedding(self, gs: GraphsData, x: Array, stmt: int) -> torch.Tensor:
        (beg, end) = gs.node_ranges[stmt]
        return torch.max(x[beg:end], dim=0, keepdims=True)[0]


class Dataset(dataset.DatasetMixin):
    def __init__(self, names: List[str], h5f) -> None:
        super().__init__()
        self._name_to_id = {name: i for (i, name) in enumerate(names)}
        self._h5f = h5f
        self._len = int(len(self._h5f["examples_conjecture"])) if "examples_conjecture" in self._h5f else 0

    def init_db(self) -> None:
        self._h5f.create_dataset("examples_conjecture", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str),
                                 compression="gzip")
        self._h5f.create_dataset("examples_statement", (0,), maxshape=(None,), dtype=np.int32, compression="gzip")

    def add_file(self, name: str, fname: Union[Path, str]) -> None:
        df = holstep.read_file(fname)
        grp = self._h5f.create_group(name)

        grp_conjecture = grp.create_group("conjecture")
        self._set_graph(grp_conjecture, self._build_graph(df.conjecture.text))

        grp.create_dataset("labels", data=np.array(df.labels, dtype=np.bool), compression="gzip")

        grp_statements = grp.create_group("statements")
        for (i, s) in enumerate(df.examples):
            grp_statement = grp_statements.create_group("%05d" % i)
            self._set_graph(grp_statement, self._build_graph(s.text))

        n = len(self._h5f["examples_conjecture"])
        self._h5f["examples_conjecture"].resize((n + len(df.examples),))
        for i in range(n, n + len(df.examples)):
            self._h5f["examples_conjecture"][i] = name
        self._h5f["examples_statement"].resize((n + len(df.examples),))
        self._h5f["examples_statement"][n:] = np.arange(len(df.examples), dtype=np.int32)
        self._len += len(df.examples)

    def _set_graph(self, grp: h5py.Group, g: GraphData) -> None:
        grp.create_dataset("labels", data=g.labels, compression="gzip")
        grp.create_dataset("edges", data=g.edges, compression="gzip")
        grp.create_dataset("treelets", data=g.treelets, compression="gzip")

    def _get_graph(self, grp: h5py.Group) -> GraphData:
        return GraphData(
            labels=grp["labels"],
            edges=grp["edges"],
            treelets=grp["treelets"],
        )

    def __len__(self) -> int:
        return self._len

    def get_example(self, i: int) -> Tuple[GraphData, GraphData, bool]:
        name = self._h5f["examples_conjecture"][i]
        j = self._h5f["examples_statement"][i]
        grp = self._h5f[name]
        g_conj = self._get_graph(grp["conjecture"])
        g_stmt = self._get_graph(grp["statements"]["%05d" % j])
        label = grp["labels"][j]
        return g_conj, g_stmt, label

    def _symbol_to_id(self, sym: str) -> int:
        if re.fullmatch(r'_\d+', sym):
            sym = "_"
        elif re.fullmatch(r'GEN%PVAR%\d+', sym):
            sym = "GEN%PVAR"
        if sym not in self._name_to_id:
            sym = "UNKNOWN"
        return self._name_to_id[sym]

    def _build_graph(self, text: str) -> GraphData:
        tokens = parser_funcparselib.tokenize(text)
        g = tree.tree_to_graph(tree.thm_to_tree(parser_funcparselib.thm.parse(tokens)))
        labels, edges, treelets = g
        return GraphData(
            labels=np.array([self._symbol_to_id(l) for l in labels], dtype=np.int32),
            edges=np.array(edges, dtype=np.int32),
            treelets=np.array(treelets, dtype=np.int32),
        )


def convert(minibatch: List[Tuple[GraphData, GraphData, bool]]) -> Tuple[GraphsData, List[Tuple[int, int, bool]]]:
    node_offset = 0
    node_ranges = []  # type: List[Tuple[int,int]]
    edge_offset = 0
    treelet_offset = 0

    table = {}  # type: Dict[int,int]

    labels = []
    edges = []
    treelets = []

    MI_data = []
    MI_row = []
    MI_col = []
    MO_data = []
    MO_row = []
    MO_col = []
    ML_data = []
    ML_row = []
    ML_col = []
    MH_data = []
    MH_row = []
    MH_col = []
    MR_data = []
    MR_row = []
    MR_col = []

    def f(gd: GraphData) -> int:
        nonlocal node_offset
        nonlocal edge_offset
        nonlocal treelet_offset
        if id(gd) in table:
            return table[id(gd)]

        labels.append(gd.labels)

        nv = len(gd.labels)

        out_edges = [[] for _ in range(nv)]  # type: List[List[int]]
        in_edges = [[] for _ in range(nv)]  # type: List[List[int]]
        for (i, (u, v)) in enumerate(gd.edges):
            out_edges[u].append(i)
            in_edges[v].append(i)

        treeletsL = [[] for _ in range(nv)]  # type: List[List[int]]
        treeletsH = [[] for _ in range(nv)]  # type: List[List[int]]
        treeletsR = [[] for _ in range(nv)]  # type: List[List[int]]
        for (i, (u, v, w)) in enumerate(gd.treelets):
            treeletsL[u].append(i)
            treeletsH[v].append(i)
            treeletsR[w].append(i)

        edges.append(np.array(gd.edges) + node_offset)
        for v in range(len(gd.labels)):
            den = (len(in_edges[v]) + len(out_edges[v]))
            for e in in_edges[v]:
                MI_data.append(1.0 / den)
                MI_row.append(node_offset + v)
                MI_col.append(edge_offset + e)
            for e in out_edges[v]:
                MO_data.append(1.0 / den)
                MO_row.append(node_offset + v)
                MO_col.append(edge_offset + e)

        treelets.append(np.array(gd.treelets) + node_offset)
        for v in range(len(gd.labels)):
            den = len(treeletsL[v]) + len(treeletsH[v]) + len(treeletsR[v])
            if den == 0:
                continue
            for t in treeletsL[v]:
                ML_data.append(1.0 / den)
                ML_row.append(node_offset + v)
                ML_col.append(treelet_offset + t)
            for t in treeletsH[v]:
                MH_data.append(1.0 / den)
                MH_row.append(node_offset + v)
                MH_col.append(treelet_offset + t)
            for t in treeletsR[v]:
                MR_data.append(1.0 / den)
                MR_row.append(node_offset + v)
                MR_col.append(treelet_offset + t)

        ret = len(node_ranges)
        node_ranges.append((node_offset, node_offset + len(gd.labels)))

        node_offset += len(gd.labels)
        edge_offset += len(gd.edges)
        treelet_offset += len(gd.treelets)

        table[id(gd)] = ret

        return ret

    minibatch2 = [(f(conj), f(stmt), y) for (conj, stmt, y) in minibatch]

    MI = torch.sparse.FloatTensor(
        torch.LongTensor([MI_row, MI_col]),
        torch.FloatTensor(MI_data),
        torch.Size((node_offset, edge_offset)))
    MO = torch.sparse.FloatTensor(
        torch.LongTensor([MO_row, MO_col]),
        torch.FloatTensor(MO_data),
        torch.Size((node_offset, edge_offset)))

    ML = torch.sparse.FloatTensor(
        torch.LongTensor([ML_row, ML_col]),
        torch.FloatTensor(ML_data),
        torch.Size((node_offset, treelet_offset)))
    MH = torch.sparse.FloatTensor(
        torch.LongTensor([MH_row, MH_col]),
        torch.FloatTensor(MH_data),
        torch.Size((node_offset, treelet_offset)))
    MR = torch.sparse.FloatTensor(
        torch.LongTensor([MR_row, MR_col]),
        torch.FloatTensor(MR_data),
        torch.Size((node_offset, treelet_offset)))

    gs = GraphsData(
        node_ranges=node_ranges,
        labels=torch.from_numpy(np.concatenate(labels)).to(torch.int64),
        edges=torch.from_numpy(np.concatenate(edges)).to(torch.int64),
        treelets=torch.from_numpy(np.concatenate(treelets)).to(torch.int64),
        MI=MI,
        MO=MO,
        ML=ML,
        MH=MH,
        MR=MR,
    )

    return gs, minibatch2


def prepare_batch(
    batch: Tuple[GraphsData, List[Tuple[int, int, bool]]],
    device: torch.device = None,
    non_blocking: bool = False
) -> Tuple[Tuple[GraphsData, List[Tuple[int, int, bool]]], torch.Tensor]:
    gs, minibatch = batch
    gs = GraphsData(
        node_ranges=gs.node_ranges,
        labels=gs.labels.to(device), # , dtype=torch.int64
        edges=gs.edges.to(device),
        treelets=gs.treelets.to(device),
        MI=gs.MI.to(device),
        MO=gs.MO.to(device),
        ML=gs.ML.to(device),
        MH=gs.MH.to(device),
        MR=gs.MR.to(device),
    )
    labels = torch.tensor([1 if y else 0 for (conj, stmt, y) in minibatch])
    labels = labels.to(device)
    return (gs, minibatch), labels
