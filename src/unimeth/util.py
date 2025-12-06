from unimeth.config import vocab

from accelerate.state import PartialState
state = PartialState()

def token2seq(tokens):
    seq = [vocab[x] for x in tokens if x != -100]
    seq = ''.join(seq)
    return seq

@state.on_local_main_process
def local_print(s):
    print(s)

