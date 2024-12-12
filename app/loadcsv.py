import pandas as pd

def loadAndReturnCsv():

    df = pd.read_csv("../knowledge_base.csv")

    trigger_responses = dict(zip(df['Trigger Phrase'], df['Response']))

    return trigger_responses





