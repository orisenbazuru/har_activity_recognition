'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
from pyseqlab.attributes_extraction import GenericAttributeExtractor

class SensorAttributeExtractorCateg(GenericAttributeExtractor):
    """class implementing observation functions that generates attributes from observations"""

    def __init__(self):
        num_sensor_channels = 113
        self.track_names = ["s_{}".format(i) for i in range(num_sensor_channels)]
        attr_desc = self.generate_attributes_desc()
        super().__init__(attr_desc)
    
    
    def generate_attributes_desc(self):
        attr_desc = {}
        for track_attr_name in self.track_names:
            attr_desc[track_attr_name] = {'description': 'sensor {}'.format(track_attr_name),
                                          'encoding':'categorical'}
        return(attr_desc)
            
    def generate_attributes(self, seq, boundaries):
        X = seq.X  
        observed_attrnames = self.track_names
        # segment attributes dictionary
        self.seg_attr = {}
        new_boundaries = []
        # create segments from observations using the provided boundaries
        for boundary in boundaries:
            if(boundary not in seq.seg_attr):
                self._create_segment(X, boundary, observed_attrnames)
                new_boundaries.append(boundary)
        if(self.seg_attr):
            # save generated attributes in seq
            seq.seg_attr.update(self.seg_attr)
            # clear the instance variable seg_attr
            self.seg_attr = {}
        return(new_boundaries)    
    
        
def example():
    from pyseqlab.utilities import DataFileParser
    import os
    current_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    dataset_dir = os.path.join(root_dir, "dataset")
    # initialize a data file parser
    dparser = DataFileParser()
    # provide the options to parser such as the header info, the separator between words and if the y label is already existing
    # main means the header is found in the first line of the file
    header = ["s_{}".format(i) for i in range(113)] + ['locomotion']
    # y_ref is a boolean indicating if the label to predict is already found in the file
    y_ref = True
    # spearator between the observations
    column_sep = " "
    seqs = []
    # read only one sequence
    for seq in dparser.read_file(os.path.join(dataset_dir, 'locomotion', 'train_discretized_iob.txt'), header, y_ref=y_ref, column_sep = column_sep):
        seqs.append(seq)
        break
    attr_extractor = SensorAttributeExtractorCateg()
    print("attr_desc {}".format(attr_extractor.attr_desc))
    attr_extractor.generate_attributes(seq, seq.get_y_boundaries())
    for boundary, seg_attr in seq.seg_attr.items():
        print("boundary {}".format(boundary))
        print("attributes {}".format(seg_attr))
    print("seg_attr {}".format(seq.seg_attr))
    return(seq)
if __name__ == "__main__":
    example()