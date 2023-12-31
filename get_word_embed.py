from sklearn.cluster import KMeans
import numpy as np
import pickle
import torch
import nltk
import clip


if __name__ == '__main__':


    ### CLIP based word embedding
    vocab = pickle.load(open('./vocab.pkl', 'rb'))
    concepts = []
    print(vocab)
    for k in vocab.itos[0:]:
        concepts.append(k)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    bsz = 100
    concept_features = []
    for i in range(0, len(concepts), bsz):
        text = concepts[i: i + bsz]
        text = clip.tokenize(text).to(device)
        with torch.no_grad():
            concept_feature = model.encode_text(text)
        concept_features.append(concept_feature.cpu())
    concept_features = torch.cat(concept_features,dim=0)

    torch.save({'clip_embeds':concept_features}, './word_embeds.pth' )


