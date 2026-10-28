# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: neuroTools
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import os

csvMainDir = "/Users/karl/Dropbox/UMD/audioMixes/mfaWork/"
subdirs = ["american/female/", "american/male/", "british/female/", "british/male/"]

for subdir in subdirs:
    for i in range(16):
        targetCsv = pd.read_csv(
            csvMainDir + "outputs/" + subdir + "testing/testing/" + str(i).zfill(2) + ".csv"
        )
        targetCsv["gpt2Word"] = 0
        targetCsv["gpt2Probability"] = 0
        targetCsv["gpt2Surprisal"] = 0
        gpt2Probs = pd.read_csv(csvMainDir + "inputs/american/female/" + str(i).zfill(2) + ".lab_readable.csv")  # This loads this from american/female because that's the only place I put it, but it's just about the text, not the speaker so it's the same for all speakers
        wordInd = 0
        for index, row in targetCsv.iterrows():
            if row["Type"] == "words":
                if gpt2Probs["Word"][wordInd] == "s" or gpt2Probs["Word"][wordInd] == "s." or gpt2Probs["Word"][wordInd] == "t" or gpt2Probs["Word"][wordInd] == "d" or gpt2Probs["Word"][wordInd] == "re" or gpt2Probs["Word"][wordInd] == "ve" or gpt2Probs["Word"][wordInd] == "m":
                    wordInd += 1
                if wordInd > 0 and gpt2Probs["Word"][wordInd - 1] == " o" and gpt2Probs["Word"][wordInd] == " the":
                    wordInd += 1
                targetCsv["gpt2Word"][index] = gpt2Probs["Word"][wordInd]
                probabilityHere = gpt2Probs["Probability"][wordInd]
                targetCsv["gpt2Probability"][index] = probabilityHere
                if probabilityHere != 0:
                    targetCsv["gpt2Surprisal"][index] = -np.log2(probabilityHere)
                wordInd += 1
        os.makedirs(csvMainDir + "outputs/" + subdir + "testing/testing/testing/", exist_ok=True)
        targetCsv.to_csv(csvMainDir + "outputs/" + subdir + "testing/testing/testing/" + str(i).zfill(2) + ".csv")

for subdir in subdirs:
    for i in range(12):
        targetCsv = pd.read_csv(
            csvMainDir + "outputs/" + subdir + "testing/testing/dist-" + str(i).zfill(2) + ".csv"
        )
        targetCsv["gpt2Word"] = 0
        targetCsv["gpt2Probability"] = 0
        targetCsv["gpt2Surprisal"] = 0
        gpt2Probs = pd.read_csv(csvMainDir + "inputs/american/female/dist-" + str(i).zfill(2) + ".lab_readable.csv")  # This loads this from american/female because that's the only place I put it, but it's just about the text, not the speaker so it's the same for all speakers
        wordInd = 0
        for index, row in targetCsv.iterrows():
            if row["Type"] == "words":
                if gpt2Probs["Word"][wordInd] == "s" or gpt2Probs["Word"][wordInd] == "s." or gpt2Probs["Word"][wordInd] == "t" or gpt2Probs["Word"][wordInd] == "d" or gpt2Probs["Word"][wordInd] == "re" or gpt2Probs["Word"][wordInd] == "ve" or gpt2Probs["Word"][wordInd] == "m":
                    wordInd += 1
                if wordInd > 0 and gpt2Probs["Word"][wordInd - 1] == " o" and gpt2Probs["Word"][wordInd] == " the":
                    wordInd += 1
                targetCsv["gpt2Word"][index] = gpt2Probs["Word"][wordInd]
                probabilityHere = gpt2Probs["Probability"][wordInd]
                targetCsv["gpt2Probability"][index] = probabilityHere
                if probabilityHere != 0:
                    targetCsv["gpt2Surprisal"][index] = -np.log2(probabilityHere)
                # if row["Label"] != "2":
                wordInd += 1
        os.makedirs(csvMainDir + "outputs/" + subdir + "testing/testing/testing/", exist_ok=True)
        targetCsv.to_csv(csvMainDir + "outputs/" + subdir + "testing/testing/testing/dist-" + str(i).zfill(2) + ".csv")


