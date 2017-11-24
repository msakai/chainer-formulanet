{-# OPTIONS_GHC -Wall #-}
import Data.Attoparsec.ByteString
import Control.Monad
import qualified Data.ByteString.Char8 as B
import Text.Printf
import Holstep2

main :: IO ()
main = do
  forM_ [(1::Int)..9999] $ \i -> do
    f (printf "holstep/train/%05d" i)
  forM_ [(1::Int)..1411] $ \i -> do
    f (printf "holstep/test/%04d" i)

f :: FilePath -> IO ()
f fname = do
  putStrLn fname
  df <- readDataFile fname
  let ts = dfConjecture df : dfDependencies df ++ map fst (dfExamples df)
  forM_ ts $ \t -> do
    case parseOnly (thm <* endOfInput) (formulaText t) of
      Right _ -> return ()
      Left err -> do
        putStrLn $ B.unpack (formulaText t)
        putStrLn err
