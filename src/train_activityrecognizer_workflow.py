'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''    
import os
from pyseqlab.features_extraction import HOFeatureExtractor
from pyseqlab.ho_crf_ad import HOCRFAD, HOCRFADModelRepresentation
from pyseqlab.workflow import GenericTrainingWorkflow
from pyseqlab.utilities import TemplateGenerator, generate_datetime_str, generate_trained_model, \
                               create_directory               
from sensor_attribute_extractor import SensorAttributeExtractorCateg
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

# define relevant directories
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
dataset_dir = os.path.join(root_dir, 'dataset')

num_sensor_channels = 113
track_attr_names = ['s_{}'.format(i) for i in range(num_sensor_channels)]

def load_DataFileParser_options(target_y):
    data_parser_options = {'header':track_attr_names + [target_y],
                           'y_ref':True,
                           'column_sep':" ",
                           'seg_other_symbol':None
                           }
    return(data_parser_options)

def template_config():
    template_generator = TemplateGenerator()
    templateXY = {}
    # generating template for tracks
    for track_attr_name in track_attr_names:
        template_generator.generate_template_XY(track_attr_name, ('1-gram:2-gram', range(-3,4)), '1-state', templateXY)
        template_generator.generate_template_XY(track_attr_name, ('1-gram', range(0,1)), '2-states:3-states', templateXY)
    templateY = {'Y':()}
    return(templateXY, templateY)

def run_training(optimization_options, template_config, target_labelname, train_fname, test_fname, num_seqs=np.inf, profile=False):
    import cProfile
    if(profile):
        local_def = {'optimization_options':optimization_options,
                     'template_config':template_config,
                     'target_labelname':target_labelname,
                     'train_fname':train_fname,
                     'test_fname':test_fname,
                     'num_seqs':num_seqs
                    }
        global_def = {'train_crfs':train_crfs}
        uid = generate_datetime_str()
        profiling_dir = create_directory('profiling', root_dir)
        cProfile.runctx('train_crfs(optimization_options, template_config, target_labelname, train_fname, test_fname, num_seqs=num_seqs)',
                        global_def, local_def, filename = os.path.join(profiling_dir, "timeprofile_{}".format(uid)))
    else:
        return(train_crfs(optimization_options, template_config, target_labelname, train_fname, test_fname, num_seqs=num_seqs))

def revive_learnedmodel(model_dir, attr_extractor):
    modelparts_dir = os.path.join(model_dir, "model_parts")
    lmodel = generate_trained_model(modelparts_dir, attr_extractor)
    return(lmodel)



def get_performance(df, other_symbol=None):
    """get performance of the model using the decoded sequences
    
       Args:
           df: pandas data frame obtained using :func:`eval_decoded_file` function
           
       Keyword Args:
           other_symbol: string or None (default). In case specified, it means we are decoding sequences 
                         with labels represented in the IOB/BIO representation. 
                         The `other_symbol` represents the non entity label (i.e. the one not represented in IOB/BIO format)
    
    """
    #df = pd.read_csv(decseq_file, header=0, sep="\t")
    cols = df.columns.tolist()
    iob_repr = False
    if(type(other_symbol) != type(None)):
        iob_repr = True
        
    if(iob_repr):
        y_pred = df[cols[-1]].str.split("-", expand=True)[1]
        y_pred.loc[y_pred.isnull()] = other_symbol 
        y_pred = y_pred.astype('int')
    else:
        y_pred = df[cols[-1]]
    y_pred = y_pred.tolist()
    if(iob_repr):
        y_ref = df[cols[-2]].str.split("-", expand=True)[1]
        y_ref.loc[y_ref.isnull()] = other_symbol
        y_ref = y_ref.astype('int')
    else:
        y_ref = df[cols[-2]]
    y_ref = y_ref.tolist()
    print(classification_report(y_ref, y_pred))
    print("weighted f1:")
    print(f1_score(y_ref, y_pred, average='weighted'))
    print("micro f1:")
    print(f1_score(y_ref, y_pred, average='micro'))
    
def eval_models(models_dir):
    train_fname = "train_fold_0.txt"
    test_fname = "test_fold_0.txt"
    for model_dir in models_dir:  
        for f in (train_fname, test_fname):
            print(f)
            file_path = os.path.join(model_dir, "decoding_seqs", f)
            eval_decoded_file(file_path, other_symbol='0')

def eval_decoded_file(fpath, sep="\t", other_symbol=None):
    """evaluate performance using a decoded file
    
       Args:
           fpath: string, the path to the decoded file

       Keyword Args:
           sep: string, columns separator
           other_symbol: string or None (default). In case specified, it means we are decoding sequences 
                         with labels represented in the IOB/BIO representation. 
                         The `other_symbol` represents the non entity label (i.e. the one not represented in IOB/BIO format)
    
    
    """
    # define colnames
    colnames = track_attr_names + ['y_ref', 'y_pred']
    # define coltypes
    coltypes = {sname:'int' for sname in track_attr_names}
    coltypes['y_ref'] = 'object'
    coltypes['y_pred'] = 'object'
    df = pd.read_csv(fpath, sep=sep, engine='c', header=None, names=colnames, dtype=coltypes)
    get_performance(df, other_symbol = other_symbol)

  
def train_crfs(optimization_options, template_config, target_labelname, train_fname, test_fname, num_seqs=np.inf):
    crf_model = HOCRFAD 
    model_repr = HOCRFADModelRepresentation
    # init attribute extractor
    sensor_attrextractor = SensorAttributeExtractorCateg()
    # get the attribute description 
    attr_desc = sensor_attrextractor.attr_desc
    
    # load templates
    template_XY, template_Y = template_config()
    # init feature extractor
    fextractor = HOFeatureExtractor(template_XY, template_Y, attr_desc)
    # no need for feature filter
    fe_filter = None

    # create a root directory for processing the data
    wd = create_directory('wd', root_dir)
    workflow_trainer = GenericTrainingWorkflow(sensor_attrextractor, fextractor, fe_filter, 
                                               model_repr, crf_model,
                                               wd)
    

    # define the data split strategy
    # since data is already organized in separate files (train and test)
    # we use all data in training file 
    dsplit_options = {'method':"none"}
    # since we are not going to train using perceptron, we can safely setup full_parsing=False
    full_parsing = False

    # load data_parser_options
    data_parser_options = load_DataFileParser_options(target_labelname)
    # load train file
    train_file = os.path.join(dataset_dir, target_labelname, train_fname)
    data_split = workflow_trainer.seq_parsing_workflow(dsplit_options,
                                                       seq_file=train_file,
                                                       data_parser_options=data_parser_options,
                                                       num_seqs=num_seqs,
                                                       full_parsing = full_parsing)
    trainseqs_id = data_split[0]['train']
    crf_m = workflow_trainer.build_crf_model(trainseqs_id,
                                             "f_0", 
                                             full_parsing=full_parsing)
    model_dir = workflow_trainer.train_model(trainseqs_id, 
                                             crf_m, 
                                             optimization_options)
    use_options_seqsinfo = {'seqs_info':workflow_trainer.seqs_info,
                            'model_eval':False,
                            'file_name':'train_fold_0.txt',
                            'sep':' '
                           }
    model_train_perf = workflow_trainer.use_model(model_dir, use_options_seqsinfo)  
    print(model_train_perf)
        
    # get test file
    test_file = os.path.join(dataset_dir, target_labelname, test_fname)
    # define the options to use the model while providing a sequence file
    use_options_seqfile = {'seq_file':test_file,
                           'data_parser_options':data_parser_options,
                           'num_seqs':num_seqs,
                           'model_eval':False,
                           'file_name':'test_fold_0.txt',
                           'sep':' '
                          }

    model_test_perf = workflow_trainer.use_model(model_dir, use_options_seqfile)
    print(model_test_perf)
    
    return(model_dir)

if __name__ == "__main__":
    pass
    
    
