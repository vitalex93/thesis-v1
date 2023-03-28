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
    entities = []
    for ent in doc.ents:
        if ent.label_ in labels and ent in ruler(doc).ents:
            entities.append(ent.text)
    return entities

def get_entities_by_dict(text, labels, model, patterns, label_dict):
    """Get a dictionary of named entities in the given `Doc` object that match the labels in the input dictionary."""
    entities_dict = {}
    for key, labels in label_dict.items():
        entities = get_entities_by_label(text, labels, model, patterns)
        entities_dict[key] = entities
    return entities_dict





