from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler


class HistoryCallback(BaseCallbackHandler):

    def __init__(self):
        self.full_history = []

    def on_llm_start(self, ooutputs, **kwargs):
        if 'response' in ooutputs:
            self.full_history.append(ooutputs['response'])
