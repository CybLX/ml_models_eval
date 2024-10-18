#%%
from nltk.corpus import stopwords
class Vocabulary:
    def __init__(self,freq_threshold, target, stopword):
        
        #initiate the index to token dict
        ## <PAD> -> padding, used for padding the shorter sentences in a batch to match the length of longest sentence in the batch
        ## <SOS> -> start token, added in front of each sentence to signify the start of sentence
        ## <EOS> -> End of sentence token, added to the end of each sentence to signify the end of sentence
        ## <UNK> -> words which are not found in the vocab are replace by this token
        
        self.stop_word = stopwords.words('portuguese') if stopword == True else None

        self.itos = {0 : "<pad>",
                    1: "<sos>",
                    2: "<eos>",
                    3: "<unk>"}
        self.soti = {k:j for j,k in self.itos.items()} 
        self.freq = None
        self.threshold = freq_threshold
        self.target = target
        self.size_vocab = None
        #self.max_size = max_size

    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, index):
        token = self.soti[index]

        return 
    @staticmethod
    def tokenizer(text):
        return text.split()


    #output ex. for stoi -> {'the':5, 'a':6, 'an':7}
    def build_vocabulary(self,sentence_list):
        freq = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                if word not in freq.keys():
                    freq[word] = 1
                else:
                    freq[word] += 1

        # remove palavras de baixa frequencia
        freq = {k:v for k,v in freq.items() if v>self.threshold} 
        if self.stop_word is not None:
            freq = {k:v for k,v in freq.items() if k not in self.stop_word}
        
        # idx =4 p/ pad, start, end , unk
        #freq = dict(sorted(freq.items(), key = lambda x: -x[1])[:self.max_size-idx]) 
        
        for word in freq.keys():
            if word not in self.soti.keys():
                self.soti[word] = idx
                self.itos[idx] = word
                idx+=1
        self.freq = freq
    
    def extract_vocab(self,data):    
        print('\n'+"*"*20+' CREATING VOCABULARY '+"*"*20, flush = True)
        self.build_vocabulary(sentence_list = data[self.target].tolist())
        self.size_vocab = len(self.soti)
        print(f'Scale of {self.target} vocabulary: {self.size_vocab}',flush = True)

    def numericalize(self,text):
        tokenize_text = self.tokenizer(text)

        return [
                self.soti[token] if token in self.soti else self.soti["<unk>"]
                for token in tokenize_text
        ]


# %%
