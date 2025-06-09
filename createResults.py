import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from graph import Graph
import ast

if __name__ == "__main__":
    resultsPath = os.path.join(os.path.dirname(__file__), "results")
    testset = os.path.join(os.path.dirname(__file__), "ds_verifier", "Dominating Set Verifier", "src", "test", "resources", "testset")



    listOfFolders = [os.path.join(resultsPath,folderName) for folderName in os.listdir(resultsPath)]
    print(listOfFolders)
    for folder in listOfFolders:
        listDirs = [os.path.join(folder,dirName) for dirName in os.listdir(folder)]
        fullGraph ={}
        for dir in listDirs:

            if "barplot.png" not in  dir:
                df = pd.read_csv(os.path.join(dir,"data.csv"))

                osSeparator = os.path.sep
                expectedSolution  = os.path.join(testset,folder.split(osSeparator)[-1].split(".")[0] + ".sol")
                expectedGraph = os.path.join(testset,folder.split(osSeparator)[-1].split(".")[0] + ".gr")

                with open(expectedSolution, "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("p") or line.startswith("c"):

                            lines.remove(line)
                    nrOfNodes = int(lines[0].strip())
                    expectedSolution = [int(line.strip()) for line in lines[1:]]


                plt.clf()
                fig, ax = plt.subplots()
                ax.bar(df["ID"], df["Time"])

                ax.set_title("Execution time")
                ax.set_xlabel("Run ID")
                ax.set_ylabel("Execution time (s)")

                plt.savefig(os.path.join(dir,"barplot.png"))
                plt.clf()
                DS =list(set( df["Solution"]))
                counter = 0
                for solution in DS:
                    try:

                        solution = ast.literal_eval(solution)
                        print(f"Solution: {solution}, len(solution): {len(solution)}")
                        print(f"Expected Solution: {expectedSolution}, len(expectedSolution): {len(expectedSolution)}")


                        if len(solution) == len(expectedSolution):
                            plt.clf()

                            counter += 1
                            plt.clf()

                        else:
                            plt.clf()

                            counter += 1
                            plt.clf()
                        dfMean = sum(df["Time"])/len(df["Time"])
                        fullGraph[dir.split(osSeparator)[-1]] = dfMean
                    except TypeError as e:
                        print(e)
                        raise e
                    except ValueError as e:
                        print(e)
                        raise e
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Solution: {solution}")

                        raise e

        print(f"fullGraph: {fullGraph}")

        plt.clf()
        fig, ax = plt.subplots()
        ax.bar(fullGraph.keys(), fullGraph.values())


        ax.set_title("Execution time")
        ax.set_xlabel("Run ID")
        ax.set_ylabel("Execution time (s)")


        plt.savefig(os.path.join(folder,"barplot.png"))

        plt.clf()

