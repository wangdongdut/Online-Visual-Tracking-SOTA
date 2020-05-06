# Online-Visual-Tracking-SOTA

This page focuses on watching the state-of-the-art performance for the short-term tracking task (if you are interested in the long-term tracking task, please visit [here](https://github.com/wangdongdut/Long-term-Visual-Tracking)). The evaluation datasets include: 
LaSOT, VOT2019, VOT2018, TrackingNet, GOT-10k, NFS, UAV123, TC-128, OTB-100. 

* **TOP-One Performance on All Datasets:**

     | LaSOT    | VOT2019 | VOT2018 | TrackingNet | Got-10k  |   NFS    | UAV123  | TC-128   | OTB-100 |
     |:--------:|:-------:|:-------:|:-----------:|:-----------:|:--------:|:-------:|:--------:|:-----------:|
     | Success  |   EAO   |   EAO   |   Success   | Success  | Success  | Success | Success  | Success |
     |  0.648   |   0.395 |  0.489  |     0.812   |  0.649   |  0.639   |  0.680  |  0.649   |  0.712 |

* **LaSOT:**

     | Tracker                   | Success Score    | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | Siam R-CNN (CVPR20)       | 0.648  |  5 (Tesla V100)   |   [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
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
     | SPLT (ICCV19)             | 0.426  |  26 (GTX 1080Ti)   |   [Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Skimming-Perusal_Tracking_A_Framework_for_Real-Time_and_Robust_Long-Term_Tracking_ICCV_2019_paper.pdf)/[Code](https://github.com/iiau-tracker/SPLT) |
     | MDNet (CVPR16)            | 0.397  |  5 (GTX 1080Ti)    |   [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Nam_Learning_Multi-Domain_Convolutional_CVPR_2016_paper.pdf)/[Code](https://github.com/hyeonseobnam/py-MDNet) |

    * MDNet is the best tracker in the original [LaSOT](https://cis.temple.edu/lasot/) paper. 

* **VOT2019:**   

     | Tracker                   | EAO    | Accuracy (A) | Robustness (R) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | DRNet (VOT2019)           | 0.395  |  0.605  |  0.261  | [Code](https://github.com/ShuaiBai623/DRNet)|

    * DRNet is the best tracker in the original [VOT2019](http://prints.vicos.si/publications/375) report. 
    
 * **VOT2018:**  
    
     | Tracker                   | EAO    | Accuracy (A) | Robustness (R) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | D3S (CVPR20)              | 0.489  |  0.640  | 0.150   | [Paper](https://arxiv.org/pdf/1911.08862.pdf)/[Code](https://github.com/alanlukezic/d3s) | 
     | SiamAttn (CVPR20)         | 0.470  |  0.63   | 0.16    | [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | MAML-Retina (CVPR20)      | 0.452  |  0.604  | 0.159   | [Paper](https://arxiv.org/pdf/2004.00830.pdf)/[Code]() | 
     | SiamBAN (CVPR20)          | 0.452  |  0.597  | 0.178   | [Paper](https://arxiv.org/pdf/2003.06761.pdf)/[Code](https://github.com/hqucv/siamban) |  
     | PrDiMP50 (CVPR20)         | 0.442  |  0.618  | 0.165   | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | DiMP50 (ICCV19)           | 0.440  |  0.587  |  0.153  | [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | SiamFC++GoogLeNet (AAAI20)| 0.426  |  0.587  |  0.183  | [Paper](https://arxiv.org/pdf/1911.06188.pdf)/[Code](https://github.com/MegviiDetection/video_analyst) |
     | SiamRPN++ (CVPR19)        | 0.414  |  0.600  | 0.234   | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/STVIR/pysot) |     
     | Siam R-CNN (CVPR20)       | 0.408  |  0.597  |  0.220  | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |    
     | ATOM (CVPR19)             | 0.401  |  0.590  |  0.204  | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Danelljan_ATOM_Accurate_Tracking_by_Overlap_Maximization_CVPR_2019_paper.pdf)/[Code](https://github.com/visionml/pytracking) |
     | LADCF (VOT2018)           | 0.389  |  0.503  |  0.159  | [Code](https://github.com/XU-TIANYANG/LADCF) |
    
    * VOT2018 and VOT2017 have same sequences. VOT2013 to VOT2016 are small-scale and out-of-date. 
    * LADCF is the best tracker in the original [VOT2018](http://prints.vicos.si/publications/365) report. 

* **TrackingNet:**

     | Tracker                   | Success Score    | Norm Precision Score | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | Siam R-CNN (CVPR20)       | 0.812  | 0.854   | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | PrDiMP50 (CVPR20)         | 0.758  | 0.816   | 30 (Unkown GPU)  | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | MAML-FCOS (CVPR20)        |  0.757 | 0.822   | 42 (NVIDIA P100) | [Paper](https://arxiv.org/pdf/2004.00830.pdf)/[Code]() |
     | SiamAttn (CVPR20)         | 0.752  | 0.817   | 45 (RTX 2080Ti)  | [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | DiMP50 (ICCV19)           | 0.740  | 0.801   |  43 (GTX 1080)   | [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | SiamRPN++ (CVPR19)        | 0.733  | 0.800   | 35 (Titan XP)    | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/STVIR/pysot) |
     
     * The performance on TrackingNet is improved very fast. Here merely list the trackers performing better than **SiamRPN++**. 
     * TrackingNet Leaderboard：http://eval.tracking-net.org/web/challenges/challenge-page/39/leaderboard

* **GOT-10k:**

     | Tracker                   | Success Score (AO) | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | Siam R-CNN (CVPR20)       | 0.649  | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | PrDiMP50 (CVPR20)         | 0.634  | 30 (Unkown GPU)  | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | DiMP50 (ICCV19)           | 0.611  | 43 (GTX 1080)    | [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | D3S (CVPR20)              | 0.597  | 25 (GTX 1080)    | [Paper](https://arxiv.org/pdf/1911.08862.pdf)/[Code](https://github.com/alanlukezic/d3s) | 
     | ATOM (CVPR19)             | 0.556  | 30 (GTX 1080)    | [Paper](https://arxiv.org/pdf/1811.07628.pdf)/[Code](https://github.com/visionml/pytracking)  |
     
     * The performance on GOT-10k has been improved significantly after ATOM. Here merely list the trackers performing better than **ATOM**. 
     * GOT-10k leaderboard: http://got-10k.aitestunion.com/leaderboard

* **NFS:**

     | Tracker                   | Success Score    | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | Siam R-CNN (CVPR20)       | 0.639  | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | PrDiMP50 (CVPR20)         | 0.635  | 30 (Unkown GPU)  | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | DiMP50 (ICCV19)           | 0.620  | 43 (GTX 1080)    | [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |

* **UAV123:**

     | Tracker                   |   Success Score  | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | PrDiMP50 (CVPR20)         | 0.680  | 30 (Unkown GPU)  | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | DiMP50 (ICCV19)           | 0.654  | 43 (GTX 1080)    | [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | Siam R-CNN (CVPR20)       | 0.649  | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | ATOM (CVPR19)             | 0.643  |  30 (GTX 1080)   | [Paper](https://arxiv.org/pdf/1811.07628.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | SiamRPN++ (CVPR19)        | 0.642  |  35 (Titan XP)   | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/STVIR/pysot) |

* **TC-128:**

     | Tracker                   |   Success Score  | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | Siam R-CNN (CVPR20)       | 0.649  | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) | 
     | UPDT (ECCV2018)           | 0.622  |      N/A         | [Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Goutam_Bhat_Unveiling_the_Power_ECCV_2018_paper.pdf)  |

* **OTB-100/OTB-2015:**
     | Tracker                   | Success Score    | Precision Score | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | SiamAttn (CVPR20)         | 0.712  | 0.926  | 45 (RTX 2080Ti)  | [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | UPDT (ECCV2018)           | 0.702  | 0.931  |      N/A         | [Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Goutam_Bhat_Unveiling_the_Power_ECCV_2018_paper.pdf)          |
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

## All Short-Term Tracking Datasets:
* **List:**

     | Datasets       | #videos    | #total/min/max/average frames|Absent Label| 
     |:-----------    |:----------------:|:----------------:|:----------------:|
     | [LaSOT](https://cis.temple.edu/lasot/)          | 1,400 (I-all-1,400/II-test-280)      |      3.52M/1,000/11,397/2,506 |  Yes |   
     | [VOT2018]()                                     | |    |  No  |
     | [TrackingNet]()                                 | |    |  No  |
     | [GOT-10k](http://got-10k.aitestunion.com/)      | train-10k, val-180, test-180 |      1.5M      |  No |  
     | [NfS](http://ci2cv.net/nfs/index.html)          | 100            |      383K/169/20,665/3,830      |  No |    
     | [UAV123](https://uav123.org/)                   | 123            |      113K/109/3,085/915      |  No |       
     | [TC-128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html)         | 128     | 55K/71/3,872/429   |  No |    
     | [OTB-2015](http://cvlab.hanyang.ac.kr/tracker_benchmark/)       | 100            |      59K/71/3,872/590       |  No | 
     | | | |  
     | [ALOV300++](http://alov300pp.joomlafree.it/)    | 315            |      8.9K/XXXX/XXXX/284      |  No |
     | [NUS-PRO](https://www.ece.nus.edu.sg/lv/pro/nus_pro.html)        | 365            |      135K/146/5,040/371      |  No |       
     | [TB70]()                     | 70           |      XXXX     |  No |         

     * [OTB-2013/OTB-50](http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html) is a subset of OTB-2015. 


## Conference Tracking Papers: 
* **2020:**

* **2019:**
     * Unsupervised Deep Tracking. CVPR, 2019.  <br /> Ning Wang, Yibing Song, Chao Ma, Wengang Zhou, Wei Liu, Houqiang Li. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Unsupervised_Deep_Tracking_CVPR_2019_paper.pdf)]

     * Fast Online Object Tracking and Segmentation: A Unifying Approach. CVPR, 2019.  <br /> Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, Philip H.S. Torr. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Fast_Online_Object_Tracking_and_Segmentation_A_Unifying_Approach_CVPR_2019_paper.pdf)] 
     
     * Object Tracking by Reconstruction With View-Specific Discriminative Correlation Filters. CVPR, 2019.  <br /> Ugur Kart, Alan Lukezic, Matej Kristan, Joni-Kristian Kamarainen, Jiri Matas. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kart_Object_Tracking_by_Reconstruction_With_View-Specific_Discriminative_Correlation_Filters_CVPR_2019_paper.pdf)] 
     
     * Target-Aware Deep Tracking. CVPR, 2019.  <br /> Xin Li, Chao Ma, Baoyuan Wu, Zhenyu He, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Target-Aware_Deep_Tracking_CVPR_2019_paper.pdf)] 
     
     * SPM-Tracker: Series-Parallel Matching for Real-Time Visual Object Tracking. CVPR, 2019.  <br />  Guangting Wang, Chong Luo, Zhiwei Xiong, Wenjun Zeng. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_SPM-Tracker_Series-Parallel_Matching_for_Real-Time_Visual_Object_Tracking_CVPR_2019_paper.pdf)] 
     
     * SiamRPN++: Evolution of Siamese Visual Tracking With Very Deep Networks. CVPR, 2019.  <br />  Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)] 
     
     * Deeper and Wider Siamese Networks for Real-Time Visual Tracking. CVPR, 2019.  <br />  Zhipeng Zhang, Houwen Peng. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deeper_and_Wider_Siamese_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf)] 
     
     * Graph Convolutional Tracking. CVPR, 2019.  <br /> Junyu Gao, Tianzhu Zhang, Changsheng Xu. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Graph_Convolutional_Tracking_CVPR_2019_paper.pdf)] 
     
     * ATOM: Accurate Tracking by Overlap Maximization. CVPR, 2019.  <br /> Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, Michael Felsberg. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Danelljan_ATOM_Accurate_Tracking_by_Overlap_Maximization_CVPR_2019_paper.pdf)] 
     
     * Visual Tracking via Adaptive Spatially-Regularized Correlation Filters. CVPR, 2019.  <br />  Kenan Dai, Dong Wang, Huchuan Lu, Chong Sun, Jianhua Li. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Visual_Tracking_via_Adaptive_Spatially-Regularized_Correlation_Filters_CVPR_2019_paper.pdf)] 
     
     * LaSOT: A High-Quality Benchmark for Large-Scale Single Object Tracking. CVPR, 2019.  <br /> Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, Haibin Ling. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Fan_LaSOT_A_High-Quality_Benchmark_for_Large-Scale_Single_Object_Tracking_CVPR_2019_paper.pdf)] 
     
     * ROI Pooled Correlation Filters for Visual Tracking. CVPR, 2019.  <br /> Yuxuan Sun, Chong Sun, Dong Wang, You He, Huchuan Lu. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_ROI_Pooled_Correlation_Filters_for_Visual_Tracking_CVPR_2019_paper.pdf)] 
     
     * Siamese Cascaded Region Proposal Networks for Real-Time Visual Tracking. CVPR, 2019.  <br /> Heng Fan, Haibin Ling. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Fan_Siamese_Cascaded_Region_Proposal_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf)] 
     
     * 
     
. ICCV, 2019.  <br />   

. [[Paper]()]


* **2018:**

* **2017:**

* **2016:**

* **2015:**
     * Structural Sparse Tracking. CVPR, 2015.  <br /> Tianzhu Zhang, Si Liu, Changsheng Xu, Shuicheng Yan, Bernard Ghanem, Narendra Ahuja, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Zhang_Structural_Sparse_Tracking_2015_CVPR_paper.pdf)]
     
     * Reliable Patch Trackers: Robust Visual Tracking by Exploiting Reliable Patches. CVPR, 2015.  <br /> Yang Li, Jianke Zhu, Steven C.H. Hoi. [[Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Li_Reliable_Patch_Trackers_2015_CVPR_paper.pdf)] 
     
     * MUlti-Store Tracker (MUSTer): A Cognitive Psychology Inspired Approach to Object Tracking. CVPR, 2015.  <br /> Zhibin Hong, Zhe Chen, Chaohui Wang, Xue Mei, Danil Prokhorov, Dacheng Tao. [[Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Hong_MUlti-Store_Tracker_MUSTer_2015_CVPR_paper.pdf)] 
     
     * In Defense of Color-Based Model-Free Tracking. CVPR, 2015.  <br /> Horst Possegger, Thomas Mauthner, Horst Bischof. [[Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Possegger_In_Defense_of_2015_CVPR_paper.pdf)] 
     
     * JOTS: Joint Online Tracking and Segmentation. CVPR, 2015.  <br /> Longyin Wen, Dawei Du, Zhen Lei, Stan Z. Li, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Wen_JOTS_Joint_Online_2015_CVPR_paper.pdf)] 
     
     * Clustering of Static-Adaptive Correspondences for Deformable Object Tracking. CVPR, 2015.  <br /> Georg Nebehay, Roman Pflugfelder. [[Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Nebehay_Clustering_of_Static-Adaptive_2015_CVPR_paper.pdf)] 
     
     * Real-Time Part-Based Visual Tracking via Adaptive Correlation Filters. CVPR, 2015.  <br /> Ting Liu, Gang Wang, Qingxiong Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Liu_Real-Time_Part-Based_Visual_2015_CVPR_paper.pdf)]
     
     * Multihypothesis Trajectory Analysis for Robust Visual Tracking. CVPR, 2015.  <br /> Dae-Youn Lee, Jae-Young Sim, Chang-Su Kim. [[Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Lee_Multihypothesis_Trajectory_Analysis_2015_CVPR_paper.pdf)] 
     
     * Long-Term Correlation Tracking. CVPR, 2015.  <br /> Chao Ma, Xiaokang Yang, Chongyang Zhang, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Ma_Long-Term_Correlation_Tracking_2015_CVPR_paper.pdf)]

     * Discriminative Low-Rank Tracking. ICCV, 2015.  <br /> Yao Sui, Yafei Tang, Li Zhang. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Sui_Discriminative_Low-Rank_Tracking_ICCV_2015_paper.pdf)] 
     
     * SOWP: Spatially Ordered and Weighted Patch Descriptor for Visual Tracking. ICCV, 2015.  <br /> Han-Ul Kim, Dae-Youn Lee, Jae-Young Sim, Chang-Su Kim. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Kim_SOWP_Spatially_Ordered_ICCV_2015_paper.pdf)] 
     
     * Multi-Kernel Correlation Filter for Visual Tracking. ICCV, 2015.  <br /> Ming Tang, Jiayi Feng. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Tang_Multi-Kernel_Correlation_Filter_ICCV_2015_paper.pdf)] 
     
     * Tracking-by-Segmentation With Online Gradient Boosting Decision Tree. ICCV, 2015.  <br /> Jeany Son, Ilchae Jung, Kayoung Park, Bohyung Han. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Son_Tracking-by-Segmentation_With_Online_ICCV_2015_paper.pdf)] 
     
     * Exploring Causal Relationships in Visual Object Tracking. ICCV, 2015.  <br /> Karel Lebeda, Simon Hadfield, Richard Bowden. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Lebeda_Exploring_Causal_Relationships_ICCV_2015_paper.pdf)] 
     
     * Hierarchical Convolutional Features for Visual Tracking. ICCV, 2015.  <br /> Chao Ma, Jia-Bin Huang, Xiaokang Yang, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Ma_Hierarchical_Convolutional_Features_ICCV_2015_paper.pdf)] 
     
     * Online Object Tracking With Proposal Selection. ICCV, 2015.  <br /> Yang Hua, Karteek Alahari, Cordelia Schmid. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Hua_Online_Object_Tracking_ICCV_2015_paper.pdf)] 
     
     * Understanding and Diagnosing Visual Tracking Systems. ICCV, 2015.  <br /> Naiyan Wang, Jianping Shi, Dit-Yan Yeung, Jiaya Jia. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Wang_Understanding_and_Diagnosing_ICCV_2015_paper.pdf)] 
     
     * Visual Tracking With Fully Convolutional Networks. ICCV, 2015.  <br /> Lijun Wang, Wanli Ouyang, Xiaogang Wang, Huchuan Lu. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Wang_Visual_Tracking_With_ICCV_2015_paper.pdf)] 
     
     * Multiple Feature Fusion via Weighted Entropy for Visual Tracking. ICCV, 2015.  <br /> Lin Ma, Jiwen Lu, Jianjiang Feng, Jie Zhou. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Ma_Multiple_Feature_Fusion_ICCV_2015_paper.pdf)] 
     
     * Local Subspace Collaborative Tracking. ICCV, 2015.  <br /> Lin Ma, Xiaoqin Zhang, Weiming Hu, Junliang Xing, Jiwen Lu, Jie Zhou. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Ma_Local_Subspace_Collaborative_ICCV_2015_paper.pdf)] 
     
     * Learning Spatially Regularized Correlation Filters for Visual Tracking. ICCV, 2015.  <br /> Martin Danelljan, Gustav Hager, Fahad Shahbaz Khan, Michael Felsberg. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Danelljan_Learning_Spatially_Regularized_ICCV_2015_paper.pdf)] 
     
     * TRIC-track: Tracking by Regression With Incrementally Learned Cascades. ICCV, 2015.  <br /> Xiaomeng Wang, Michel Valstar, Brais Martinez, Muhammad Haris Khan, Tony Pridmore. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Wang_TRIC-track_Tracking_by_ICCV_2015_paper.pdf)] 
     
     * Linearization to Nonlinear Learning for Visual Tracking. ICCV, 2015.  <br /> Bo Ma, Hongwei Hu, Jianbing Shen, Yuping Zhang, Fatih Porikli. [[Paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Ma_Linearization_to_Nonlinear_ICCV_2015_paper.pdf)] 

* **2014:**
     * Adaptive Color Attributes for Real-Time Visual Tracking. CVPR, 2014.  <br /> Martin Danelljan, Fahad Shahbaz Khan, Michael Felsberg, Joost van de Weijer. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Danelljan_Adaptive_Color_Attributes_2014_CVPR_paper.pdf)[

    * Multi-Cue Visual Tracking Using Robust Feature-Level Fusion Based on Joint Sparse Representation. CVPR, 2014.  <br /> Xiangyuan Lan, Andy J. Ma, Pong C. Yuen. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Lan_Multi-Cue_Visual_Tracking_2014_CVPR_paper.pdf)]

    * Multi-Forest Tracker: A Chameleon in Tracking. CVPR, 2014.  <br /> David J. Tan, Slobodan Ilic. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Tan_Multi-Forest_Tracker_A_2014_CVPR_paper.pdf)] 
    
    * Pyramid-based Visual Tracking Using Sparsity Represented Mean Transform. CVPR, 2014.  <br /> Zhe Zhang, Kin Hong Wong. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Zhang_Pyramid-based_Visual_Tracking_2014_CVPR_paper.pdf)] 
    
    * Partial Occlusion Handling for Visual Tracking via Robust Part Matching. CVPR, 2014.  <br /> Tianzhu Zhang, Kui Jia, Changsheng Xu, Yi Ma, Narendra Ahuja. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Zhang_Partial_Occlusion_Handling_2014_CVPR_paper.pdf)] 
    
    * Speeding Up Tracking by Ignoring Features. CVPR, 2014.  <br /> Lu Zhang, Hamdi Dibeklioglu, Laurens van der Maaten. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Zhang_Speeding_Up_Tracking_2014_CVPR_paper.pdf)] 
    
    * Online Object Tracking, Learning and Parsing with And-Or Graphs. CVPR, 2014.  <br /> Yang Lu, Tianfu Wu, Song Chun Zhu. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Lu_Online_Object_Tracking_2014_CVPR_paper.pdf)] 
    
    * Visual Tracking via Probability Continuous Outlier Model. CVPR, 2014.  <br /> Dong Wang, Huchuan Lu. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Wang_Visual_Tracking_via_2014_CVPR_paper.pdf)]  
    
    * Visual Tracking Using Pertinent Patch Selection and Masking. CVPR, 2014.  <br /> Dae-Youn Lee, Jae-Young Sim, Chang-Su Kim. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Lee_Visual_Tracking_Using_2014_CVPR_paper.pdf)]  
    
    * Interval Tracker: Tracking by Interval Analysis. CVPR, 2014.  <br /> Junseok Kwon, Kyoung Mu Lee. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kwon_Interval_Tracker_Tracking_2014_CVPR_paper.pdf)]  
    
    * Unifying Spatial and Attribute Selection for Distracter-Resilient Tracking. CVPR, 2014.  <br /> Nan Jiang, Ying Wu. [[Paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Jiang_Unifying_Spatial_and_2014_CVPR_paper.pdf)]

* **2013:**
     * Visual Tracking via Locality Sensitive Histograms. CVPR, 2013.  <br /> Shengfeng He, Qingxiong Yang, Rynson W.H. Lau, Jiang Wang, Ming-Hsuan Yang. 
       
     * Online Object Tracking: A Benchmark. CVPR, 2013.  <br /> Yi Wu, Jongwoo Lim, Ming-Hsuan Yang. 
       
     * Learning Compact Binary Codes for Visual Tracking. CVPR, 2013.  <br /> Xi Li, Chunhua Shen, Anthony Dick, Anton van den Hengel. 
       
     * Least Soft-Threshold Squares Tracking. CVPR, 2013.  <br /> Dong Wang, Huchuan Lu, Ming-Hsuan Yang. 
       
     * Part-Based Visual Tracking with Online Latent Structural Learning. CVPR, 2013.  <br /> Rui Yao, Qinfeng Shi, Chunhua Shen, Yanning Zhang, Anton van den Hengel.
       
     * Minimum Uncertainty Gap for Robust Visual Tracking. CVPR, 2013.  <br /> Junseok Kwon, Kyoung Mu Lee.
       
     * Structure Preserving Object Tracking. CVPR, 2013.  <br /> Lu Zhang, Laurens van der Maaten. 
       
     * Self-Paced Learning for Long-Term Tracking. CVPR, 2013.  <br /> James S. Supancic III, Deva Ramanan. 
       
     * Tracking via Robust Multi-task Multi-view Joint Sparse Representation. ICCV, 2013.  <br /> Zhibin Hong, Xue Mei, Danil Prokhorov, Dacheng Tao. 
       
     * Online Robust Non-negative Dictionary Learning for Visual Tracking. ICCV, 2013.  <br /> Naiyan Wang, Jingdong Wang, Dit-Yan Yeung. 
       
     * Robust Object Tracking with Online Multi-lifespan Dictionary Learning. ICCV, 2013.   <br /> Junliang Xing, Jin Gao, Bing Li, Weiming Hu, Shuicheng Yan. 
       
     * Finding the Best from the Second Bests - Inhibiting Subjective Bias in Evaluation of Visual Tracking Algorithms. ICCV, 2013.    <br /> Yu Pang, Haibin Ling. 
       
     * PixelTrack: A Fast Adaptive Algorithm for Tracking Non-rigid Objects. ICCV, 2013. <br /> Stefan Duffner, Christophe Garcia. 
       
     * Discriminant Tracking Using Tensor Representation with Semi-supervised Improvement.ICCV, 2013. <br /> Jin Gao, Junliang Xing, Weiming Hu, Steve Maybank. 
       
     * Initialization-Insensitive Visual Tracking through Voting with Salient Local Features. ICCV, 2013. <br /> Kwang Moo Yi, Hawook Jeong, Byeongho Heo, Hyung Jin Chang, Jin Young Choi. 
       
     * Randomized Ensemble Tracking. ICCV, 2013. <br /> Qinxun Bai, Zheng Wu, Stan Sclaroff, Margrit Betke, Camille Monnier. 
       
     * Tracking Revisited Using RGBD Camera: Unified Benchmark and Baselines. ICCV, 2013. <br /> Shuran Song, Jianxiong Xiao. 
       
     * Modeling Self-Occlusions in Dynamic Shape and Appearance Tracking. ICCV, 2013. <br /> Yanchao Yang, Ganesh Sundaramoorthi. 
       
     * Orderless Tracking through Model-Averaged Posterior Estimation. ICCV, 2013. <br /> Seunghoon Hong, Suha Kwak, Bohyung Han. 
     
* **2012:**

* **2011:**

* **2010:**

* **2009:**

* **2008:**

* **2007:**

* **2006:**

* **2005:**

* **2004:**

* **2003:**

* **2002:**

* **2001:**

* **2000:**
