import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler

nlp = spacy.blank("en")
#ruler = EntityRuler(nlp)
ruler = nlp.add_pipe('entity_ruler')

patterns = [{"label": "APPTYPE", "pattern":  [{'LOWER': 'application'},{'LOWER': 'type'}]},
            {"label": "DCA", "pattern": [{"LOWER": "per", 'OP':'?'}, {"LOWER": "dca"}]}]

ruler.add_patterns(patterns)
#nlp.add_pipe(ruler, before="ner")

Q7 = 'Create a report that shows the number of settlement applications approved during the month, their approved amount, the written off balance,\
      the average days to approval, their average and median duration and their entry principal and balance,\
          for September 2020 DCA and application type. The report should be produced on Earth portfolio.'
doc = nlp(Q7)
for ent in doc.ents:
    print(ent.text, ent.label_)

displacy.render(doc, style='ent')




