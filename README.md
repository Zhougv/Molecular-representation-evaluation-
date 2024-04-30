# Molecular-representation-evaluation-
Molecular representation evaluation is all you need
We evaluated 25 types of representation methods on 17 benchmark datasets. Unifined score is used as our evaluation index. The overall evaluation results are as follows: Figure 1 shows the results of each representation on each dataset and Figure 2 shows the overall evaluation of each representation based on the results on each dataset.
![b76f5e5473ee517486391df09534886](https://github.com/Zhougv/Molecular-representation-evaluation-/assets/164281953/734ebb06-1199-4be3-902a-ea44b0fef3de)
Fig1  Unifined score of each representation on each dataset
![adde218e125ec3c117e16229cedc779](https://github.com/Zhougv/Molecular-representation-evaluation-/assets/164281953/22765d0e-bde7-4a64-ba4a-4d832195212f)
Fig2  The overall evaluation of each representation based on the results on each dataset

The 17 data sets evaluated can be found in the dataset directory. For the 12 fingerprint codes in the representation method and the code of the machine learning algorithm you can find in the ML&fingerprint directory. We provide github links to 13 deep learn-based representations. We made simple modifications to it to accommodate predictions for different tasks.

BAN  
Learning to SMILES: BAN-based strategies to improve latent representation learning from molecules 
https://github.com/zhang-xuan1314/SMILES-BiLSTM-attention-network

MolMapNet
Out-of-the-box deep learning prediction of pharmaceutical properties by broadly learned knowledge-based 
https://github.com/shenwanxiang/bidd-molmap

AttentiveFP
Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism
https://github.com/OpenDrugAI/AttentiveFP

N-Gram Graph
N-Gram Graph: Simple Unsupervised Representation for Graphs, with Applications to Molecules
https://github.com/chao1224/n_gram_graph

TrimNet
TrimNet: learning molecular representation from triplet messages for biomedicine
https://github.com/yvquanli/trimnet

MolCLR
Molecular contrastive learning of representations via graph neural networks
https://github.com/yuyangw/MolCLR

ImageMol
Accurate prediction of molecular properties and drug targets using a self-supervised image representation learning framework
https://github.com/ChengF-Lab/ImageMol

SphereNet
SPHERICAL MESSAGE PASSING FOR 3D MOLECULAR GRAPHS
https://github.com/divelab/DIG

GraphMVP
PRE-TRAINING MOLECULAR GRAPH REPRESENTATION WITH 3D GEOMETRY
https://github.com/chao1224/GraphMVP

3d Infomax
3D INFOMAX IMPROVES GNNS FOR MOLECULAR PROPERTY PREDICTION
https://github.com/HannesStark/3DInfomax

Finally, we provide a user-friendly web platform for displaying, storing all the data sets involved in this study, results, and supporting users to make relevant property predictions of new compounds on this platform.
http://59.110.25.27:8501/

