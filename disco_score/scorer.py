from disco_score.metrics import discourse
from disco_score.metrics.word_embeddings import load_embeddings
from transformers import BertConfig, BertTokenizer, BertModel
import torch

class DiscoScorer: 

	def __init__(self, device='cuda:0', model_name='bert-base-uncased', we=None, truncation=None):

		config = BertConfig.from_pretrained(model_name, output_hidden_states=True, output_attentions=True, return_dict=True)
		self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
		self.model = BertModel.from_pretrained(model_name, config=config)
		self.model.encoder.layer = torch.nn.ModuleList([layer for layer in self.model.encoder.layer[:8]])
		self.model.eval()
		self.model.to(device)  
		if we is not None:
		    we = load_embeddings('deps', we) 		
		self.we = we
		self.truncation = truncation

	def LexicalChain(self, sys, ref):
	    return discourse.LexicalChain(sys, ref)
	    
	def DS_Focus_NN(self, sys, ref):
	    sys = self.__truncate(sys)
	    ref = self.__truncate(ref)
	    return discourse.DS_Focus(self.model, self.tokenizer, sys, ref, is_semantic_entity=False)

	def DS_Focus_Entity(self, sys, ref):
	    sys = self.__truncate(sys)
	    ref = self.__truncate(ref)
	    return discourse.DS_Focus(self.model, self.tokenizer, sys, ref, is_semantic_entity=True, we=self.we, threshold = 0.8)

	def DS_SENT_NN(self, sys, ref):
	    sys = self.__truncate(sys)
	    ref = self.__truncate(ref)
	    return discourse.DS_Sent(self.model, self.tokenizer, sys, ref, is_lexical_graph=False)

	def DS_SENT_Entity(self, sys, ref):
	    sys = self.__truncate(sys)
	    ref = self.__truncate(ref)
	    return discourse.DS_Sent(self.model, self.tokenizer, sys, ref, is_lexical_graph=True, we=self.we, threshold = 0.5)
	        
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
	        return disco_scorer.tokenizer.convert_tokens_to_string(tokens[:min(truncate - 1, len(tokens))])
	    else:
	        return text
