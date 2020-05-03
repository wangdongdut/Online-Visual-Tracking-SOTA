# Online-Visual-Tracking-SOTA

This page focuses on watching the state-of-the-art performance for the short-term tracking task (if you are interested in the long-term tracking task, please visit [here](https://github.com/wangdongdut/Long-term-Visual-Tracking)). The evaluation datasets include: 
LaSOT, VOT2019, TrackingNet, GOT-10k, NFS, UAV123, TC-128, OTB-100. 

* **LaSOT:**

     | Tracker                   | Success Score    | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | Siam R-CNN (CVPR20)       | 0.648  |  5 (Tesla V100) |   [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | PrDiMP50 (CVPR20)         | 0.598  |  30 (Unkown GPU)  |   [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | LTMU (CVPR20)             | 0.572  |  13 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2004.00305)/[Code](https://github.com/Daikenan/LTMU) |
     | DiMP50 (ICCV19)           | 0.568  |  43 (GTX 1080)    |   [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | SiamAttn (CVPR20)         | 0.560  |  45 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | SiamFC++GoogLeNet (AAAI20)| 0.544  |  90 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/1911.06188.pdf)/[Code](https://github.com/MegviiDetection/video_analyst) |
     | MAML-FCOS (CVPR20)        | 0.523  |  42 (NVIDIA P100) |   [Paper](https://arxiv.org/pdf/2004.00830.pdf)/[Code]() |
     | GlobalTrack (AAAI20)      | 0.521  |  6 (GTX TitanX)   |   [Paper](https://arxiv.org/abs/1912.08531)/[Code](https://github.com/huanglianghua/GlobalTrack) |
     | ATOM (CVPR19)             | 0.515  |  30 (GTX 1080)    |   [Paper](https://arxiv.org/pdf/1811.07628.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | SiamBAN (CVPR20)          | 0.514  |  40 (GTX 1080Ti)  |   [Paper](https://arxiv.org/pdf/2003.06761.pdf)/[Code](https://github.com/hqucv/siamban) |  
     | SiamCar (CVPR20)          | 0.507  |  52 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/1911.07241.pdf)/[Code](https://github.com/ohhhyeahhh/SiamCAR) |   
     | SiamRPN++ (CVPR19)        | 0.496  |  35 (Titan XP)    |   [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/STVIR/pysot) |
     | ROAM++ (CVPR20)           | 0.447  |  20 (RTX 2080)|  [Paper](https://arxiv.org/pdf/1907.12006.pdf)/[Code](https://github.com/skyoung/ROAM) |
     | SPLT (ICCV19)             | 0.426  |  26 (GTX 1080Ti)       |      [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Skimming-Perusal_Tracking_A_Framework_for_Real-Time_and_Robust_Long-Term_Tracking_ICCV_2019_paper.pdf)/[Code](https://github.com/iiau-tracker/SPLT) |
     | MDNet (CVPR16)            | 0.397  |  5 (GTX 1080Ti)       | [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Nam_Learning_Multi-Domain_Convolutional_CVPR_2016_paper.pdf)/[Code](https://github.com/hyeonseobnam/py-MDNet) |

    * MDNet is the best tracker in the original [LaSOT](https://cis.temple.edu/lasot/) paper. 

* **VOT2019/VOT2018:**   

     | Tracker                   | EAO    | Accuracy (A) | Robustness (R) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | SiamAttn (CVPR20)         | 0.470  |  0.63   | 0.16    | [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | MAML-Retina (CVPR20)      | 0.452  |  0.604  | 0.159   |  [Paper](https://arxiv.org/pdf/2004.00830.pdf)/[Code]() | 
     | SiamBAN (CVPR20)          | 0.452  |  0.597  | 0.178   |  [Paper](https://arxiv.org/pdf/2003.06761.pdf)/[Code](https://github.com/hqucv/siamban) |  
     | PrDiMP50 (CVPR20)         | 0.442  |  0.618  | 0.165   | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | DiMP50 (ICCV19)           | 0.440  |  0.587  |  0.153  | [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | SiamFC++GoogLeNet (AAAI20)| 0.426  |  0.587  |  0.183  | [Paper](https://arxiv.org/pdf/1911.06188.pdf)/[Code](https://github.com/MegviiDetection/video_analyst) |
     | SiamRPN++ (CVPR19)        | 0.414  |  0.600  | 0.234   | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/STVIR/pysot) |     
     | Siam R-CNN (CVPR20)       | 0.408  |  0.597  |  0.220  | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |    
     | ATOM (CVPR19)             | 0.401  |  0.590  |  0.204  | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Danelljan_ATOM_Accurate_Tracking_by_Overlap_Maximization_CVPR_2019_paper.pdf)/[Code](https://github.com/visionml/pytracking) |
     | DRNet (VOT2019)           | 0.395  |  0.605  |  0.261  | [Code](https://github.com/ShuaiBai623/DRNet)|

    * DRNet is the best tracker in the original [VOT2019](http://prints.vicos.si/publications/375) report. 
    * VOT2019 and VOT2018 have same sequences. VOT2013 to VOT2017 are small-scale and out-of-date. 

* **TrackingNet:**

     | Tracker                   | Success Score    | Norm Precision Score | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | SiamAttn (CVPR20)         | 0.752  | 0.817  | 45 (RTX 2080Ti)  | [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | Siam R-CNN (CVPR20)       | 0.701  | 0.891  | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | DRT (CVPR18)              | 0.699  | 0.923  |        N/A       | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Correlation_Tracking_via_CVPR_2018_paper.pdf)/[Code](https://github.com/cswaynecool/DRT) |
     | PrDiMP50 (CVPR20)         | 0.696  |  N/A   | 30 (Unkown GPU)  | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | SiamRPN++ (CVPR19)        | 0.696  | 0.914  | 35 (Titan XP)    | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/STVIR/pysot) |
     | MCCT (CVPR18)             | 0.696  | 0.914  | 8 (GTX 1080Ti)   | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Multi-Cue_Correlation_Filters_CVPR_2018_paper.pdf)/[Code](https://github.com/594422814/MCCT) |
     | SiamBAN (CVPR20)          | 0.696  | 0.910  | 40 (GTX 1080Ti)  | [Paper](https://arxiv.org/pdf/2003.06761.pdf)/[Code](https://github.com/hqucv/siamban) | 
     | GFS-DCF (ICCV19)          | 0.693  | 0.932  |  8 (Titan X)     | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Joint_Group_Feature_Selection_and_Discriminative_Filter_Learning_for_Robust_ICCV_2019_paper.pdf)/[Code](https://github.com/XU-TIANYANG/GFS-DCF) |    
     | SACF (ECCV18)             | 0.693  | 0.917  | 23 (GTX Titan)   | [Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/mengdan_zhang_Visual_Tracking_via_ECCV_2018_paper.pdf)|
     | ASRCF(CVPR19)             | 0.692  | 0.922  | 28 (GTX 1080Ti)  | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Visual_Tracking_via_Adaptive_Spatially-Regularized_Correlation_Filters_CVPR_2019_paper.pdf)/[Code](https://github.com/Daikenan/ASRCF) |
     | LSART (CVPR18)            | 0.691  | 0.923  |  1 (Titan X)     | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Learning_Spatial-Aware_Regressions_CVPR_2018_paper.pdf)/[Code](https://github.com/cswaynecool/LSART) |
     | ECO (ICCV17)              | 0.691  | N/A  |    8 (Unkown GPU)  | [Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Danelljan_ECO_Efficient_Convolution_CVPR_2017_paper.pdf)/[M-Code](https://github.com/martin-danelljan/ECO)/[P-Code](https://github.com/visionml/pytracking)|

* **GOT-10k:**

* **NFS:**

* **UAV123:**

* **TC-128:**

* **OTB-100/OTB-2015:**
     | Tracker                   | Success Score    | Precision Score | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | SiamAttn (CVPR20)         | 0.712  | 0.926  | 45 (RTX 2080Ti)  | [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | UPDT (ECCV2018)           | 0.702  | 0.931  |      N/A            | [Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Goutam_Bhat_Unveiling_the_Power_ECCV_2018_paper.pdf)          |
     | Siam R-CNN (CVPR20)       | 0.701  | 0.891  | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | DRT (CVPR18)              | 0.699  | 0.923  |        N/A       | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Correlation_Tracking_via_CVPR_2018_paper.pdf)/[Code](https://github.com/cswaynecool/DRT) |
     | PrDiMP50 (CVPR20)         | 0.696  |  N/A   | 30 (Unkown GPU)  | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | SiamRPN++ (CVPR19)        | 0.696  | 0.914  | 35 (Titan XP)    | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/STVIR/pysot) |
     | MCCT (CVPR18)             | 0.696  | 0.914  | 8 (GTX 1080Ti)   | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Multi-Cue_Correlation_Filters_CVPR_2018_paper.pdf)/[Code](https://github.com/594422814/MCCT) |
     | SiamBAN (CVPR20)          | 0.696  | 0.910  | 40 (GTX 1080Ti)  | [Paper](https://arxiv.org/pdf/2003.06761.pdf)/[Code](https://github.com/hqucv/siamban) | 
     | GFS-DCF (ICCV19)          | 0.693  | 0.932  |  8 (Titan X)     | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Joint_Group_Feature_Selection_and_Discriminative_Filter_Learning_for_Robust_ICCV_2019_paper.pdf)/[Code](https://github.com/XU-TIANYANG/GFS-DCF) |    
     | SACF (ECCV18)             | 0.693  | 0.917  | 23 (GTX Titan)   | [Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/mengdan_zhang_Visual_Tracking_via_ECCV_2018_paper.pdf)|
     | ASRCF(CVPR19)             | 0.692  | 0.922  | 28 (GTX 1080Ti)  | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Visual_Tracking_via_Adaptive_Spatially-Regularized_Correlation_Filters_CVPR_2019_paper.pdf)/[Code](https://github.com/Daikenan/ASRCF) |
     | LSART (CVPR18)            | 0.691  | 0.923  |  1 (Titan X)     | [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Learning_Spatial-Aware_Regressions_CVPR_2018_paper.pdf)/[Code](https://github.com/cswaynecool/LSART) |
     | ECO (ICCV17)              | 0.691  | N/A  |    8 (Unkown GPU)  | [Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Danelljan_ECO_Efficient_Convolution_CVPR_2017_paper.pdf)/[M-Code](https://github.com/martin-danelljan/ECO)/[P-Code](https://github.com/visionml/pytracking)|

    * Too many trackers are tested on OTB-100, here merely list the SOTA trackers whose success scores are larger than **0.690**. 
    * OTB-50 and OTB-2013 are similar subsets of OTB-100. 
    * It seems that the OTB-100 dataset has already been overfitting. 
