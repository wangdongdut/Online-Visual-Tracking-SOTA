# Online-Visual-Tracking-SOTA

This page focuses on watching the state-of-the-art performance for the short-term tracking task (if you are interested in the long-term tracking task, please visit [here](https://github.com/wangdongdut/Long-term-Visual-Tracking)). The evaluation datasets include: 
LaSOT, VOT2019, TrackingNet, GOT-10k, NFS, UAV123, TC-128, OTB-100. 

* **LaSOT:**

     | Tracker                   | Success Score    | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | Siam R-CNN (CVPR20)       | 0.648  |  5 (Tesla V100) |   [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | PrDiMP50 (CVPR20)         | 0.598  |  30 (Unkown GPU)  |   [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | **LTMU (CVPR20)**         | 0.572  |  13 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2004.00305)/[Code](https://github.com/Daikenan/LTMU) |
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
     | **SPLT (ICCV19)**         | 0.426  |  26 (GTX 1080Ti)       |      [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Skimming-Perusal_Tracking_A_Framework_for_Real-Time_and_Robust_Long-Term_Tracking_ICCV_2019_paper.pdf)/[Code](https://github.com/iiau-tracker/SPLT) |
     | MDNet (CVPR16)            | 0.397  |  5 (GTX 1080Ti)       | [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Nam_Learning_Multi-Domain_Convolutional_CVPR_2016_paper.pdf)/[Code](https://github.com/hyeonseobnam/py-MDNet) |

    * MDNet is the best tracker in the original [LaSOT](https://cis.temple.edu/lasot/) paper. 

* **VOT2019:**

* **TrackingNet:**

* **GOT-10k:**

* **NFS:**

* **UAV123:**

* **TC-128:**

* **OTB-100/OTB-2015:**
     | Tracker                   | Success Score    | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | SiamAttn (CVPR20)         | 0.712  |  45 (RTX 2080Ti)  | [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | UPDT (ECCV2018)           | 0.702  |                   |           |
     | Siam R-CNN (CVPR20)       | 0.701  |  5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | DRT (CVPR18)              | 0.699  | 
     | PrDiMP50 (CVPR20)         | 0.696  |  30 (Unkown GPU)  | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | SiamRPN++ (CVPR19)        | 0.696  |  35 (Titan XP)    | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/STVIR/pysot) |
     | SiamBAN (CVPR20)          | 0.696  |  40 (GTX 1080Ti)  | [Paper](https://arxiv.org/pdf/2003.06761.pdf)/[Code](https://github.com/hqucv/siamban) |  
     | ASRCF(CVPR19)             | 0.692  |                   | [Paper]()[Code]()|
     | ECO ()                    | 0.691  |                   | [Paper]()[Code]()|
     | DiMP50 (ICCV19)           | 0.687  |  30 (TITAN X)     | [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Bhat_Learning_Discriminative_Model_Prediction_for_Tracking_ICCV_2019_paper.pdf)[Code](https://github.com/visionml/pytracking)|
     | MDNet (CVPR16)            | 0.678  |  5 (GTX 1080Ti)       | [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Nam_Learning_Multi-Domain_Convolutional_CVPR_2016_paper.pdf)/[Code](https://github.com/hyeonseobnam/py-MDNet) |
     | RT-MDNet (ECCV18)         | 0.650   | | |
