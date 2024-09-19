Here is the source code for training and evaluating the proposed approach for the AutoPET III competition. In this study, a two-stage deep learning approach was developed for lesion segmentation in PET/CT images as part of the AutoPET-III challenge. The method involved using different neural network architectures for each stage, with an ensemble of models applied in the second stage to produce the final segmentation, benefiting from both coarse-to-fine segmentation refinement and the diverse strengths of multiple deep learning architectures. Additionally, the list of trained weights of the model is shared here:

Stage 1:
DynUnet: https://drive.google.com/file/d/1I-44jLRH57Ij2jUQ_pPhy9TKq8_FUgLB/view?usp=sharing 

Stage 2:
SegResNet: https://drive.google.com/file/d/1H9qoq22ispgEprW5-wwJq08o6SzLhuPE/view?usp=sharing
SwinUnetr: https://drive.google.com/file/d/1BIrRqHr3Q_pxS8cvxz8U3-DUQSLjcFsw/view?usp=sharing
UNet: https://drive.google.com/file/d/1AYmplOLJAg4edgLHBam5DxEdW1bS5Wa2/view?usp=sharing
