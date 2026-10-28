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
import pandas as pd
import numpy as np
import glob
import csv


# %%
mainDir = "/Users/karl/Library/CloudStorage/Box-Box/box-group-simon-neural-data/Cocktail-Party-EEG-MEG/behavior/SPIN/"

conditions = [
    "FemaleBrit_over_FemaleAmer_Easy",
    "MaleBrit___Quiet",
    "FemaleAmer_over_FemaleBrit_Hard",
    "MaleAmer___over_MaleBrit_Easy",
    "FemaleAmer_Quiet",
    "MaleAmer___over_FemaleAmer_Easy",
    "FemaleBrit_over_FemaleAmer_Hard",
    "MaleBrit___over_MaleAmer_Easy",
    "FemaleAmer_over_MaleAmer_Hard",
    "MaleAmer___Quiet",
    "FemaleAmer_over_MaleAmer_Easy",
    "MaleBrit___over_MaleAmer_Hard",
    "FemaleBrit_Quiet",
    "MaleAmer___over_MaleBrit_Hard",
    "FemaleAmer_over_FemaleBrit_Easy",
    "MaleAmer___over_FemaleAmer_Hard",
]

difficulties = {
    "FemaleBrit_over_FemaleAmer_Easy": "Easy",
    "MaleBrit___Quiet": "Quiet",
    "FemaleAmer_over_FemaleBrit_Hard": "Hard",
    "MaleAmer___over_MaleBrit_Easy": "Easy",
    "FemaleAmer_Quiet": "Quiet",
    "MaleAmer___over_FemaleAmer_Easy": "Easy",
    "FemaleBrit_over_FemaleAmer_Hard": "Hard",
    "MaleBrit___over_MaleAmer_Easy": "Easy",
    "FemaleAmer_over_MaleAmer_Hard": "Hard",
    "MaleAmer___Quiet": "Quiet",
    "FemaleAmer_over_MaleAmer_Easy": "Easy",
    "MaleBrit___over_MaleAmer_Hard": "Hard",
    "FemaleBrit_Quiet": "Quiet",
    "MaleAmer___over_MaleBrit_Hard": "Hard",
    "FemaleAmer_over_FemaleBrit_Easy": "Easy",
    "MaleAmer___over_FemaleAmer_Hard": "Hard",
}

genders = {
    "FemaleBrit_over_FemaleAmer_Easy": "Same",
    "MaleBrit___Quiet": "Single",
    "FemaleAmer_over_FemaleBrit_Hard": "Same",
    "MaleAmer___over_MaleBrit_Easy": "Same",
    "FemaleAmer_Quiet": "Single",
    "MaleAmer___over_FemaleAmer_Easy": "Mixed",
    "FemaleBrit_over_FemaleAmer_Hard": "Same",
    "MaleBrit___over_MaleAmer_Easy": "Same",
    "FemaleAmer_over_MaleAmer_Hard": "Mixed",
    "MaleAmer___Quiet": "Single",
    "FemaleAmer_over_MaleAmer_Easy": "Mixed",
    "MaleBrit___over_MaleAmer_Hard": "Same",
    "FemaleBrit_Quiet": "Single",
    "MaleAmer___over_MaleBrit_Hard": "Same",
    "FemaleAmer_over_FemaleBrit_Easy": "Same",
    "MaleAmer___over_FemaleAmer_Hard": "Mixed",
}

gendersAll = {
    "FemaleBrit_over_FemaleAmer_Easy": "F_F",
    "MaleBrit___Quiet": "Single",
    "FemaleAmer_over_FemaleBrit_Hard": "F_F",
    "MaleAmer___over_MaleBrit_Easy": "M_M",
    "FemaleAmer_Quiet": "Single",
    "MaleAmer___over_FemaleAmer_Easy": "M_F",
    "FemaleBrit_over_FemaleAmer_Hard": "F_F",
    "MaleBrit___over_MaleAmer_Easy": "M_M",
    "FemaleAmer_over_MaleAmer_Hard": "F_M",
    "MaleAmer___Quiet": "Single",
    "FemaleAmer_over_MaleAmer_Easy": "F_M",
    "MaleBrit___over_MaleAmer_Hard": "M_M",
    "FemaleBrit_Quiet": "Single",
    "MaleAmer___over_MaleBrit_Hard": "M_M",
    "FemaleAmer_over_FemaleBrit_Easy": "F_F",
    "MaleAmer___over_FemaleAmer_Hard": "M_F",
}

targetAccent = {
    "FemaleBrit_over_FemaleAmer_Easy": "British",
    "MaleBrit___Quiet": "British",
    "FemaleAmer_over_FemaleBrit_Hard": "American",
    "MaleAmer___over_MaleBrit_Easy": "American",
    "FemaleAmer_Quiet": "American",
    "MaleAmer___over_FemaleAmer_Easy": "American",
    "FemaleBrit_over_FemaleAmer_Hard": "British",
    "MaleBrit___over_MaleAmer_Easy": "British",
    "FemaleAmer_over_MaleAmer_Hard": "American",
    "MaleAmer___Quiet": "American",
    "FemaleAmer_over_MaleAmer_Easy": "American",
    "MaleBrit___over_MaleAmer_Hard": "British",
    "FemaleBrit_Quiet": "British",
    "MaleAmer___over_MaleBrit_Hard": "American",
    "FemaleAmer_over_FemaleBrit_Easy": "American",
    "MaleAmer___over_FemaleAmer_Hard": "American",
}

distractorAccent = {
    "FemaleBrit_over_FemaleAmer_Easy": "American",
    "MaleBrit___Quiet": "Single",
    "FemaleAmer_over_FemaleBrit_Hard": "British",
    "MaleAmer___over_MaleBrit_Easy": "British",
    "FemaleAmer_Quiet": "Single",
    "MaleAmer___over_FemaleAmer_Easy": "American",
    "FemaleBrit_over_FemaleAmer_Hard": "American",
    "MaleBrit___over_MaleAmer_Easy": "American",
    "FemaleAmer_over_MaleAmer_Hard": "American",
    "MaleAmer___Quiet": "Single",
    "FemaleAmer_over_MaleAmer_Easy": "American",
    "MaleBrit___over_MaleAmer_Hard": "American",
    "FemaleBrit_Quiet": "Single",
    "MaleAmer___over_MaleBrit_Hard": "British",
    "FemaleAmer_over_FemaleBrit_Easy": "British",
    "MaleAmer___over_FemaleAmer_Hard": "American",
}

snr = {
    "FemaleBrit_over_FemaleAmer_Easy": "4dB",
    "MaleBrit___Quiet": "Single",
    "FemaleAmer_over_FemaleBrit_Hard": "0dB",
    "MaleAmer___over_MaleBrit_Easy": "4dB",
    "FemaleAmer_Quiet": "Single",
    "MaleAmer___over_FemaleAmer_Easy": "0dB",
    "FemaleBrit_over_FemaleAmer_Hard": "0dB",
    "MaleBrit___over_MaleAmer_Easy": "4dB",
    "FemaleAmer_over_MaleAmer_Hard": "-4dB",
    "MaleAmer___Quiet": "Single",
    "FemaleAmer_over_MaleAmer_Easy": "0dB",
    "MaleBrit___over_MaleAmer_Hard": "0dB",
    "FemaleBrit_Quiet": "Single",
    "MaleAmer___over_MaleBrit_Hard": "0dB",
    "FemaleAmer_over_FemaleBrit_Easy": "4dB",
    "MaleAmer___over_FemaleAmer_Hard": "-4dB",
}

targetGender = {
    "FemaleBrit_over_FemaleAmer_Easy": "Female",
    "MaleBrit___Quiet": "Male",
    "FemaleAmer_over_FemaleBrit_Hard": "Female",
    "MaleAmer___over_MaleBrit_Easy": "Male",
    "FemaleAmer_Quiet": "Female",
    "MaleAmer___over_FemaleAmer_Easy": "Male",
    "FemaleBrit_over_FemaleAmer_Hard": "Female",
    "MaleBrit___over_MaleAmer_Easy": "Male",
    "FemaleAmer_over_MaleAmer_Hard": "Female",
    "MaleAmer___Quiet": "Male",
    "FemaleAmer_over_MaleAmer_Easy": "Female",
    "MaleBrit___over_MaleAmer_Hard": "Male",
    "FemaleBrit_Quiet": "Female",
    "MaleAmer___over_MaleBrit_Hard": "Male",
    "FemaleAmer_over_FemaleBrit_Easy": "Female",
    "MaleAmer___over_FemaleAmer_Hard": "Male",
}

distractorGender = {
    "FemaleBrit_over_FemaleAmer_Easy": "Female",
    "MaleBrit___Quiet": "Single",
    "FemaleAmer_over_FemaleBrit_Hard": "Female",
    "MaleAmer___over_MaleBrit_Easy": "Male",
    "FemaleAmer_Quiet": "Single",
    "MaleAmer___over_FemaleAmer_Easy": "Female",
    "FemaleBrit_over_FemaleAmer_Hard": "Female",
    "MaleBrit___over_MaleAmer_Easy": "Male",
    "FemaleAmer_over_MaleAmer_Hard": "Male",
    "MaleAmer___Quiet": "Single",
    "FemaleAmer_over_MaleAmer_Easy": "Male",
    "MaleBrit___over_MaleAmer_Hard": "Male",
    "FemaleBrit_Quiet": "Single",
    "MaleAmer___over_MaleBrit_Hard": "Male",
    "FemaleAmer_over_FemaleBrit_Easy": "Female",
    "MaleAmer___over_FemaleAmer_Hard": "Female",
}


subjectsToAverage = [
    "R3045",
    "R3089",
    "R3093",
    "R3095",
    "R3157",
    "R2783",
    "R3170",
    "R3172",
    "R3193",
    "R3184",
    "R3214",
    "R3265",
    "R3275",
    "R3247",
    "R3292",
    "R3293",
    "R3294",
    "R3303",
    "R3304",
    "R3306",
    "R3313",
    "R3317",
    "R3320",
]

constantToAdd = 3

sentenceInds = np.array([1, 4, 7, 10])

mainResultsSheet = [
    [
        "Subject",
        "Condition",
        "Sentence",
        "TotalCorrect",
        "TotalMissed",
        "TotalWords",
        "Ratio",
        "dB",
        "difficulties",
        "genders",
        "gendersAll",
        "targetAccent",
        "distractorAccent",
        "targetGender",
        "distractorGender",
        "snr",
    ]
]
perConditionResultsSheet = [
    [
        "Subject",
        "Condition",
        "TotalCorrect",
        "TotalMissed",
        "TotalWords",
        "Ratio",
        "dB",
        "difficulties",
        "genders",
        "gendersAll",
        "targetAccent",
        "distractorAccent",
        "targetGender",
        "distractorGender",
        "snr",
    ]
]

for subject in subjectsToAverage:
    print(f"Starting subject {subject}")
    filename = glob.glob(mainDir + subject + "*.xlsx")[
        0
    ]  # There should definitely only be one filename per subject number, so safe to just take the first one of a list which will only have one element
    for condition in conditions:
        sheet = pd.read_excel(filename, sheet_name=condition)
        perConditionTotalCorrect = 0
        perConditionTotalWords = 0
        sentenceCount = 1
        for sentenceInd in sentenceInds:
            rowInd = 2  # The results always start at the third row
            totalCorrect = 0
            while rowInd < sheet.shape[0] and not pd.isna(
                sheet.iloc[rowInd, sentenceInd]
            ):
                if sheet.iloc[
                    rowInd, sentenceInd
                ]:  # These will be either True or False, so just treat as bool
                    totalCorrect += 1
                rowInd += 1
            totalWords = rowInd - 2
            totalMissed = totalWords - totalCorrect
            perConditionTotalCorrect += totalCorrect
            perConditionTotalWords += totalWords
            ratio = totalCorrect / totalWords
            dB = 20 * np.log10(
                (totalCorrect + constantToAdd) / (totalWords + constantToAdd)
            )
            mainResultsSheet.append(
                [
                    subject,
                    condition,
                    sentenceCount,
                    totalCorrect,
                    totalMissed,
                    totalWords,
                    ratio,
                    dB,
                    difficulties.get(condition),
                    genders.get(condition),
                    gendersAll.get(condition),
                    targetAccent.get(condition),
                    distractorAccent.get(condition),
                    targetGender.get(condition),
                    distractorGender.get(condition),
                    snr.get(condition),
                ]
            )
            sentenceCount += 1
        perConditionTotalMissed = perConditionTotalWords - perConditionTotalCorrect
        perConditionRatio = perConditionTotalCorrect / perConditionTotalWords
        perConditionDB = 20 * np.log10(
            (perConditionTotalCorrect + constantToAdd)
            / (perConditionTotalWords + constantToAdd)
        )
        perConditionResultsSheet.append(
            [
                subject,
                condition,
                perConditionTotalCorrect,
                perConditionTotalMissed,
                perConditionTotalWords,
                perConditionRatio,
                perConditionDB,
                difficulties.get(condition),
                genders.get(condition),
                gendersAll.get(condition),
                targetAccent.get(condition),
                distractorAccent.get(condition),
                targetGender.get(condition),
                distractorGender.get(condition),
                snr.get(condition),
            ]
        )

with open("mainResultsSheet.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(mainResultsSheet)

with open("perConditionResultsSheet.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(perConditionResultsSheet)
