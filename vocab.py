class Vocab(object):

    def __init__(self):
        self.token_id = {}
        self.id_token = {}
        self.UNK_ID = 0
        self.seq = 1

    def size(self):
        return len(self.token_id) + 1  # +1 for UNK

    def id_to_token(self, token, update=True):
        if token in self.token_id:
            return self.token_id[token]
        elif not update:
            return self.UNK_ID
        else:
            self.token_id[token] = self.seq
            self.id_token[self.seq] = token
            self.seq += 1
            return self.seq - 1

    def ids_for_tokens(self, tokens, update=True):
        return [self.id_to_token(t, update) for t in tokens]

    def tokens_for_ids(self, ids):
        return [self.id_to_tokens[i] for i in ids]
