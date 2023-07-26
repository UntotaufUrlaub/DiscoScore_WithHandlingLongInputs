from disco_score.metrics import discourse
from disco_score.metrics.word_embeddings import load_embeddings
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch

class DiscoScorer: 

	def __init__(self, device='cuda:0', model_name='bert-base-uncased', we=None, truncation=None):
		config = AutoConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True, return_dict=True)
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
		self.model = AutoModel.from_pretrained(model_name, config=config)
		self.model.encoder.layer = torch.nn.ModuleList([layer for layer in self.model.encoder.layer[:8]])
		self.model.eval()
		self.model.to(device)  
		if we is not None:
		    we = load_embeddings('deps', we) 		
		self.we = we
		self.truncation = truncation
		self.device = device

	def LexicalChain(self, sys, ref):
	    return discourse.LexicalChain(sys, ref)
	    
	def DS_Focus_NN(self, sys, ref):
	    sys = self.__truncate(sys)
	    for i in range(len(ref)):
	    	ref[i] = self.__truncate(ref[i])
	    return discourse.DS_Focus(self.model, self.tokenizer, sys, ref, is_semantic_entity=False, device=self.device)

	def DS_Focus_Entity(self, sys, ref):
	    sys = self.__truncate(sys)
	    for i in range(len(ref)):
	    	ref[i] = self.__truncate(ref[i])
	    return discourse.DS_Focus(self.model, self.tokenizer, sys, ref, is_semantic_entity=True, we=self.we, threshold = 0.8, device=self.device)

	def DS_SENT_NN(self, sys, ref):
	    sys = self.__truncate(sys)
	    for i in range(len(ref)):
	    	ref[i] = self.__truncate(ref[i])
	    return discourse.DS_Sent(self.model, self.tokenizer, sys, ref, is_lexical_graph=False, device=self.device)

	def DS_SENT_Entity(self, sys, ref):
	    sys = self.__truncate(sys)
	    for i in range(len(ref)):
	    	ref[i] = self.__truncate(ref[i])
	    return discourse.DS_Sent(self.model, self.tokenizer, sys, ref, is_lexical_graph=True, we=self.we, threshold=0.5, device=self.device)
	        
	def RC(self, sys, ref):
	    return discourse.RC(sys)

	def LC(self, sys, ref):
	    return discourse.LC(sys)        

	def EntityGraph(self, sys, ref):
	    adjacency_dist, num_sentences = discourse.EntityGraph(sys)
	    return adjacency_dist.sum() / num_sentences

	def LexicalGraph(self, sys, ref):
	    adjacency_dist, num_sentences = discourse.EntityGraph(sys, is_lexical_graph=True, we=self.we, threshold=0.7)
	    return adjacency_dist.sum() / num_sentences

	def __truncate(self, text):
	    if self.truncation is not None:
	        tokens = self.tokenizer.tokenize(text)
	        return self.tokenizer.convert_tokens_to_string(tokens[:min(self.truncation - 1, len(tokens))])
	    else:
	        return text
