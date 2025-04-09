import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from main import Graph, draw_graph,Logger
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
            # if os.path.exists(os.path.join(dir,"data.csv")):
            if "barplot.png" not in  dir:
                df = pd.read_csv(os.path.join(dir,"data.csv"))
                # create barplot with df[id] and df[time]
                osSeparator = os.path.sep
                expectedSolution  = os.path.join(testset,folder.split(osSeparator)[-1].split(".")[0] + ".sol")
                expectedGraph = os.path.join(testset,folder.split(osSeparator)[-1].split(".")[0] + ".gr")
                # read expected solution
                with open(expectedSolution, "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("p") or line.startswith("c"):
                    #   remove first line
                            lines.remove(line)
                    nrOfNodes = int(lines[0].strip())
                    expectedSolution = [int(line.strip()) for line in lines[1:]]

                # reset plot
                plt.clf()
                fig, ax = plt.subplots()
                ax.bar(df["ID"], df["Time"])
                # set title and labels
                ax.set_title("Execution time")
                ax.set_xlabel("Run ID")
                ax.set_ylabel("Execution time (s)")
                # save figure
                plt.savefig(os.path.join(dir,"barplot.png"))
                plt.clf()
                DS =list(set( df["Solution"]))
                counter = 0
                for solution in DS:
                    try:
                        # check if solution is equal to expected solution
                        solution = ast.literal_eval(solution)
                        print(f"Solution: {solution}, len(solution): {len(solution)}")
                        print(f"Expected Solution: {expectedSolution}, len(expectedSolution): {len(expectedSolution)}")


                        if len(solution) == len(expectedSolution):
                            plt.clf()
                            draw_graph(expectedGraph,solution,f"Graph correct {len(solution)}",os.path.join(dir,f"correct{counter}.png"))
                            counter += 1
                            plt.clf()

                        else:
                            plt.clf()
                            draw_graph(expectedGraph,solution,f"Graph incorrect {len(solution)}",os.path.join(dir,f"incorrect{counter}.png"))
                            counter += 1
                            plt.clf()
                        dfMean = sum(df["Time"])/len(df["Time"])
                        fullGraph[dir.split(osSeparator)[-1]] = dfMean
                    except TypeError as e:
                        print(e)
                        pass
                    except ValueError as e:
                        print(e)
                        pass
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Solution: {solution}")
                # save fi
                        raise e

        print(f"fullGraph: {fullGraph}")
        # create barplot with fullGraph
        plt.clf()
        fig, ax = plt.subplots()
        ax.bar(fullGraph.keys(), fullGraph.values())

        # set title and labels
        ax.set_title("Execution time")
        ax.set_xlabel("Run ID")
        ax.set_ylabel("Execution time (s)")
        # add legend
        # save figure
        plt.savefig(os.path.join(folder,"barplot.png"))
        # reset plot
        plt.clf()

