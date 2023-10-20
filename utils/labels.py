#!/usr/bin/python
#
# Cityscapes labels
#

from __future__ import print_function, absolute_import, division
from collections import namedtuple
import numpy as np

import torch

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'category'    , # The name of the category that this label belongs to

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

labels = [
    #       name                     id    category          color(RGB)
    Label(  'Beef'                  ,  0 ,  'background'     , (  0,  0,  0) ),
    Label(  'Chicken'               ,  1 ,  'background'     , ( 50, 50, 50) ),
    Label(  'Tool'                  ,  2 ,  'instrument'     , (150,150,150) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

id2color        = torch.zeros((3,3))
for label in labels:
    id = label.id
    if id >= 0:
        id2color[id] = torch.tensor(label.color)

binary_id2color = torch.tensor([[50, 50, 50], [255, 255, 255]], dtype=torch.uint8)