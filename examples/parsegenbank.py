"""parse genebank file."""
from pathlib import Path
import sys
import csv
from collections import defaultdict, Counter

from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.Graphics import GenomeDiagram
from reportlab.lib.units import cm

import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles

def convert_genbank_translation(cds_annotation):
    """ converts genebank file CDS region annotation to prediction-enabled names. """
    conversion_table = {
        'minor tail protein' : 'MinorTail',
        'major capsid subunit' : 'MajorCapsid',
        'portal protein' : 'Portal',
        'hypothetical protein' : 'Hypothetical',
    }
    return conversion_table.get(cds_annotation, cds_annotation)

def parse_genbank_file(genbank_file_path: Path):
    """ Obtain the CDS regions from a genbank file. """
    cds_regions = []
    genome_length = 0
    with open(genbank_file_path, 'r') as f:
        for record in SeqIO.parse(f, 'genbank'):
            genome_length = len(record._seq)
            for feature in record.features:
                if feature.type != "CDS":
                    continue
                cds_regions.append({
                    'genbank_start' : int(feature.location.start),
                    'genbank_end' : int(feature.location.end),
                    'genbank_annotation' : f"{feature.qualifiers["note"][0]} ({convert_genbank_translation(feature.qualifiers['product'][0])})",
                    'genbank_prediction' : convert_genbank_translation(feature.qualifiers['product'][0])
                })
    return cds_regions, genome_length

def parse_prediction_file(prediction_file_path: Path, start_index=2, end_index=3):
    """ Parse the prediction file. """
    model_names = []
    predictions = []
    with open(prediction_file_path, 'r') as f:
        prediction_csv = csv.reader(f, delimiter=',')
        for index, row in enumerate(prediction_csv):
            if index == 0:
                model_names += row[6:]
                continue
            prediction_temp = {
                'phanotate_start' : int(row[start_index]), 
                'phanotate_end' : int(row[end_index])
            }
            for ind, model_name in enumerate(model_names):
                pvp_prediction = row[6+ind] if row[6+ind] != '-1' else "Non-PVP"
                prediction_temp.update({model_name : pvp_prediction})
            predictions.append(prediction_temp)

    return predictions, model_names

def phanotate_results(cds_regions, predictions):
    """ Get the results for PHANOTATE CDS predictions. """
    already_added = set()
    for genbank_cds in cds_regions:
        start_end_tuple = (genbank_cds['genbank_start'], genbank_cds['genbank_end'] )
        if start_end_tuple in already_added: continue
        already_added.add(start_end_tuple)
    print(f"genbank unique count: {len(already_added)}")
    
    already_added = set()
    for genbank_cds in predictions:
        start_end_tuple = (genbank_cds['phanotate_start'], genbank_cds['phanotate_end'] )
        if start_end_tuple in already_added: continue
        already_added.add(start_end_tuple)
    print(f"PHANOTATE unique CDS count: {len(already_added)}")

def join_genbank_and_predictions(cds_regions, predictions, threshold=10):
    """ Joins the prediction file with genbank based on the start and end indices. """
    joined_results = []
    start_end = set()
    already_added = set()
    for pred in predictions:
        for genbank_cds in cds_regions:
            if (abs(genbank_cds['genbank_start'] - pred['phanotate_start']) > threshold) or \
               (abs(genbank_cds['genbank_end'] - pred['phanotate_end']) > threshold):
                continue
            start_end_tuple = (genbank_cds['genbank_start'], genbank_cds['genbank_end'] )
            if start_end_tuple in already_added: continue
            already_added.add(start_end_tuple)

            pred.update(genbank_cds)
            joined_results.append(pred)
            start_end.add(str(genbank_cds['genbank_start']) + ' ' + str(genbank_cds['genbank_end']))

    return joined_results

def get_prediction_color(prediction):
    """ Get a color representation for a prediction. """
    shades_of_green = {
        'MajorCapsid' : '#7cfc00',
        'MinorCapsid' : '#64ee00',
        'Baseplate' : '#4de100',
        'MajorTail' : '#36d300',
        'MinorTail' :'#1fc600',
        'Portal' : '#19b800',
        'TailFiber' : '#13ab00', 
        'Collar' : '#0d9d00',
        'Shaft' : '#089000',
        'HTJ' : '#088300',
        'Hypothetical' : '#00008B'
    }
    return shades_of_green.get(prediction, '#AA4A44')

def create_genome_map(gdd, counter, joined_results, genome_length, annotation_name):
    """Creates a figure for a given class."""
    gdt_features = gdd.new_track(counter, greytrack=False,  height=1, name=annotation_name)
    gds_features = gdt_features.new_set()

    # Add three features to show the strand options,
    for cds_region in joined_results:
        feature = SeqFeature(SimpleLocation(cds_region['genbank_start'], cds_region['genbank_end'], strand=0))
        gds_features.add_feature(feature,
                                 name=cds_region[annotation_name],
                                 label_size=10,
                                 label_angle=45,
                                 label_position="middle",
                                 sigil="OCTO",
                                 label=True,
                                 color=get_prediction_color(cds_region[annotation_name]))


if __name__ == '__main__':
    genbank_file_path = Path(sys.argv[1])
    prediction_file_path = Path(sys.argv[2])

    # parse genbank and prediction file.
    cds_regions, genome_length = parse_genbank_file(genbank_file_path)
    predictions, model_names = parse_prediction_file(prediction_file_path)
    print(f"genome length:{genome_length}")
    phanotate_results(cds_regions, predictions)
    
    # join genbank and predictions.
    joined_results = join_genbank_and_predictions(cds_regions, predictions)

    # plot results across the genome.
    gdd = GenomeDiagram.Diagram("Test Diagram")
    counter = 0
    number_of_models = len(model_names) + 1
    for annotation_name in joined_results[0].keys():
        if annotation_name not in ('genbank_start', 'genbank_end', 'phanotate_start', 'phanotate_end', 'genbank_annotation'):
            print(f"genome order {number_of_models - counter} - {annotation_name}")
            create_genome_map(gdd, counter*2, joined_results, genome_length, annotation_name)
            counter += 1
    gdd.draw(format="linear", fragments=1, start=0, end=genome_length, pagesize=(40 * cm, 35 * cm))
    gdd.write(f"all_genomemap.pdf", "PDF")

    # plot vendiagram of overlapping predictions.
    modelname2predictionset = defaultdict(Counter)
    for cds in joined_results:
        for cds_info, value in cds.items():
            if cds_info in model_names:
                modelname2predictionset[cds_info][value] += 1
    out = venn3(subsets=modelname2predictionset.values(),
          set_labels =modelname2predictionset.keys())
    venn3_circles(subsets=modelname2predictionset.values(),
                  linestyle='dashed',
                  linewidth=2,
                  color="grey")
    for text in out.subset_labels:
        text.set_fontsize(16)
    plt.savefig('venn.png', dpi=300, bbox_inches='tight')
    
    # get metrics
    true_annotations = {
        'MajorCapsid',
        'MinorCapsid',
        'Baseplate',
        'MajorTail',
        'MinorTail',
        'Portal',
        'TailFiber',
        'Collar',
        'Shaft',
        'HTJ'
    }
    metrics_per_model = {}
    for cds in joined_results:
        for cds_info, value in cds.items():
            if cds_info in model_names:
                if cds_info not in metrics_per_model:
                    metrics_per_model[cds_info] = [0,0,0,0]
                if value == cds['genbank_prediction']:
                    metrics_per_model[cds_info][0] += 1 # TP
                elif cds['genbank_prediction'] not in true_annotations and value == "Non-PVP":
                    print(cds_info, cds['genbank_prediction'], value)
                    metrics_per_model[cds_info][1] += 1 # TN
                elif cds['genbank_prediction'] in true_annotations and value == "Non-PVP":
                    metrics_per_model[cds_info][2] += 1 # FN
                elif cds['genbank_prediction'] not in true_annotations and value != "Non-PVP":
                    metrics_per_model[cds_info][3] += 1 # FP
    for model, metrics in metrics_per_model.items():
        recall = metrics[0] / (metrics[0] + metrics[2])
        precision = metrics[0] / (metrics[0] + metrics[3]) if (metrics[0] + metrics[3]) > 0 else 0
        f1_score = 2 * precision * recall /  (recall + precision) if (recall + precision) > 0 else 0
        print(f"{model}: recall: {round(recall, 2)}, precision: {round(precision, 2)}, f1 score: {round(f1_score, 2)}")
        print(f"\tTP: {metrics[0]}, FP: {metrics[3]}, TN: {metrics[1]}, FN: {metrics[2]}")
                    
                    