import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler
from helper import *



def create_ruler(patterns, model):
    """Create a spaCy rule-based `EntityRuler` object with the provided patterns."""
    nlp = model
    ruler = EntityRuler(model, overwrite_ents=True)
    if 'entity_ruler' not in nlp.pipe_names:
        ruler = nlp.add_pipe('entity_ruler')
    ruler.add_patterns(patterns)
    return ruler

def get_entities_by_label(text, labels, model, patterns):
    """Get a list of named entities in the given `Doc` object that match the specified labels."""
    ruler = create_ruler(patterns=patterns, model=model)
    nlp = model
    doc = nlp(text)
    '''entities = []
    for ent in doc.ents:
        if ent.label_ in labels and ent in ruler(doc).ents:
            entities.append(ent.text)'''
    entities = {}
    for ent in doc.ents:
        if ent.label_ in labels and ent in ruler(doc).ents:
            entities[str(ent.label_)] = ent.text
    return entities

def get_entities_by_dict(text, labels, model, patterns, label_dict):
    """Get a dictionary of named entities in the given `Doc` object that match the labels in the input dictionary."""
    entities_dict = {}
    for key, labels in label_dict.items():
        entities = get_entities_by_label(text, labels, model, patterns)
        entities_dict[key] = entities
    return entities_dict


def classify_docs(docs, model, patterns):
    nlp = model
    ruler = create_ruler(patterns=patterns, model=nlp)
    return [doc.ents[0].label_ if doc.ents else None for doc in nlp.pipe(docs)]

'''def classify_docs(docs):
    nlp = spacy.blank("en")
    ruler = EntityRuler(nlp)
    patterns = [{'label':'RowGroup4', 'pattern': [{'LOWER': 'application'},{'LOWER': 'type'}]},
                {'label':'RowGroup4', 'pattern': [{"LOWER": "per", 'OP':'?'}, {"LOWER": "dca"}]},
                {'label':'RowGroup1', 'pattern': [{'LEMMA': 'payment'}]},
                {'label':'RowGroup1', 'pattern': [{'LEMMA': 'account'}]}]
    ruler.add_patterns(patterns)
    ruler = nlp.add_pipe('entity_ruler')

    return [doc.ents[0].label_ if doc.ents else None for doc in nlp.pipe(docs)]'''






