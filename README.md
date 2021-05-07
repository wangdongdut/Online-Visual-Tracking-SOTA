# Online-Visual-Tracking-SOTA

This page focuses on watching the state-of-the-art performance for the short-term tracking task (if you are interested in the long-term tracking task, please visit **[here](https://github.com/wangdongdut/Long-term-Visual-Tracking)**). The evaluation datasets include: 
LaSOT, VOT2018, TrackingNet, GOT-10k, OTB-100, NFS, UAV123, TC-128, and VOT family.  If you are also interested in some resources on Paper Writting (computer vision), please visit **[here](https://github.com/wangdongdut/PaperWriting)**.  

**:star2: Recent SOTA Works:**  
     * Learning Spatio-Temporal Transformer for Visual Tracking. Arxiv, 2021.  <br /> Bin Yan, Houwen Peng, Jianlong Fu, Dong Wang, Huchuan Lu. [[Paper](https://arxiv.org/abs/2103.17154)][[Code](https://github.com/researchmm/Stark)] <br />
     （Peformance: LaSOT, Success 0.671; TrackingNet, Success, 0.820; VOT2020LT, F-Score, 0.701; FPS, 31.7 on Tesla V100） <br />
     * Learning Target Candidate Association to Keep Track of What Not to Track. Arxiv, 2021.  <br /> Christoph Mayer, Martin Danelljan, Danda Pani Paudel, Luc Van Gool. [[Paper](https://arxiv.org/abs/2103.16556)][[Code](https://github.com/visionml/pytracking)] <br />
     （Peformance: LaSOT, Success 0.672; OxUAVLT, MaxGM, 0.812, FPS, 12.3 on RTX2080Ti） <br />
     
* **TOP-One Performance on All Datasets:**

     | **LaSOT**  | **TrackingNet** | **Got-10k**  | **OTB-100** | NFS    | UAV123  | TC-128   |
     |:--------:|:-------:|:-------:|:-----------:|:-----------:|:--------:|:-------:|
     | Success  | Success   |   AO  |  Success | Success  | Success | Success  |
     |  0.653   | 0.814   |  0.649   |  0.715   | 0.639   |  0.691  |  0.649   |
     |  ARDiMPsuper (CVPR21)  |     TransT (CVPR21)   |  Siam R-CNN (CVPR20)  |  RPT (ECCVW20)   | TrDimp (CVPR21)  |  TransT (CVPR21) |  Siam R-CNN (CVPR20)  |

* **LaSOT:star2:**

     | Tracker                   | Success Score    | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | ARDiMPsuper (CVPR21)      | 0.653  |  33 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2012.06815)/[Code](https://github.com/MasterBin-IIAU/AlphaRefine) |
     | TransT (CVPR21)           | 0.649  |  50 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2103.15436)/[Code](https://github.com/chenxin-dlut/TransT) |
     | Siam R-CNN (CVPR20)       | 0.648  |  5 (Tesla V100)   |   [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | TrDimp (CVPR21)           | 0.639  |  26 (GTX 1080Ti)  |   [Paper](https://arxiv.org/abs/2103.11681)/[Code](https://github.com/594422814/TransformerTrack) |
     | PrDiMP50 (CVPR20)         | 0.598  |  30 (Unkown GPU)  |   [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | LTMU (CVPR20)             | 0.572  |  13 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2004.00305)/[Code](https://github.com/Daikenan/LTMU) |
     | DiMP50 (ICCV19)           | 0.568  |  43 (GTX 1080)    |   [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | Ocean (ECCV20)            | 0.560  |  25 (Tesla V100)  |   [Paper](https://arxiv.org/abs/2006.10721)/[Code](https://github.com/researchmm/TracKit) |  
     | GlobalTrack (AAAI20)      | 0.521  |  6 (GTX TitanX)   |   [Paper](https://arxiv.org/abs/1912.08531)/[Code](https://github.com/huanglianghua/GlobalTrack) |

     * **Baseline(short-term): SiamRPN++, Ocean, DiMP, PrDiMP, Siam R-CNN**
     * **Baseline(long-term): GlobalTrack, LTMU, Siam R-CNN**
    
* **TrackingNet:star2:**

     | Tracker                   | Success Score    | Norm Precision Score | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | TransT (CVPR21)           | 0.814  | 0.867   | 50 (RTX 2080Ti)  | [Paper](https://arxiv.org/abs/2103.15436)/[Code](https://github.com/chenxin-dlut/TransT) |       
     | Siam R-CNN (CVPR20)       | 0.812  | 0.854   |  5 (Tesla V100)  | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | ARDiMPsuper (CVPR21)      | 0.805  | 0.856   |  33 (RTX 2080Ti) | [Paper](https://arxiv.org/abs/2012.06815)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | TrDimp (CVPR21)           | 0.784  | 0.833   |  26 (GTX 1080Ti) | [Paper](https://arxiv.org/abs/2103.11681)/[Code](https://github.com/594422814/TransformerTrack) |
     | PrDiMP50 (CVPR20)         | 0.758  | 0.816   | 30 (Unkown GPU)  | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | DiMP50 (ICCV19)           | 0.740  | 0.801   |  43 (GTX 1080)   | [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | SiamRPN++ (CVPR19)        | 0.733  | 0.800   | 35 (Titan XP)    | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/STVIR/pysot) |
     
     * TrackingNet Leaderboard：http://eval.tracking-net.org/web/challenges/challenge-page/39/leaderboard
     * **Baseline: SiamRPN++, Ocean, DiMP, PrDiMP, Siam R-CNN**

* **GOT-10k:star2:**

     | Tracker                   | Success Score (AO) | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | Siam R-CNN (CVPR20)       | 0.649  | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | PrDiMP50 (CVPR20)         | 0.634  | 30 (Unkown GPU)  | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | RPT (ECCVW20)             | 0.624  | 20 (GTX 1080Ti)  | [Paper](https://arxiv.org/pdf/2008.03467.pdf)|      
     | Ocean (ECCV20)            | 0.611  | 25 (Tesla V100)  | [Paper](https://arxiv.org/abs/2006.10721)/[Code](https://github.com/researchmm/TracKit) |  
     | DiMP50 (ICCV19)           | 0.611  | 43 (GTX 1080)    | [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     
     * GOT-10k leaderboard: http://got-10k.aitestunion.com/leaderboard
     * **Baseline: SiamRPN++, Ocean, DiMP, PrDiMP, Siam R-CNN**

* **OTB-100/OTB-2015:star2:**
     | Tracker                   | Success Score    | Precision Score | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | RPT (ECCVW20)             | 0.715  | 0.936  | 20 (GTX 1080Ti)  | [Paper](https://arxiv.org/pdf/2008.03467.pdf)|    
     | SiamAttn (CVPR20)         | 0.712  | 0.926  | 45 (RTX 2080Ti)  | [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | TrDimp (CVPR21)           | 0.711  |        | 26 (GTX 1080Ti)  | [Paper](https://arxiv.org/abs/2103.11681)/[Code](https://github.com/594422814/TransformerTrack) |
     | UPDT (ECCV2018)           | 0.702  | 0.931  |      N/A         | [Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Goutam_Bhat_Unveiling_the_Power_ECCV_2018_paper.pdf)          |
     | Siam R-CNN (CVPR20)       | 0.701  | 0.891  | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | TransT (CVPR21)           | 0.694  | -  | 50 (RTX 2080Ti) | [Paper](https://arxiv.org/abs/2103.15436)/[Code](https://github.com/chenxin-dlut/TransT) |

    * **Baseline: SiamRPN++, Ocean, DiMP, PrDiMP, Siam R-CNN**
    * OTB-50 and OTB-2013 are similar subsets of OTB-100. 
    * It seems that the OTB-100 dataset has already been overfitting. 
   
* **NFS**

     | Tracker                   | Success Score    | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | TrDimp (CVPR21)           | 0.665  |  26 (GTX 1080Ti) | [Paper](https://arxiv.org/abs/2103.11681)/[Code](https://github.com/594422814/TransformerTrack) |
     | TransT (CVPR21)           | 0.657  |  50 (RTX 2080Ti) | [Paper](https://arxiv.org/abs/2103.15436)/[Code](https://github.com/chenxin-dlut/TransT) |
     | Siam R-CNN (CVPR20)       | 0.639  |  5 (Tesla V100)  | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
    
     * **Baseline: SiamRPN++, Ocean, DiMP, PrDiMP, Siam R-CNN**

* **UAV123**

     | Tracker                   |   Success Score  | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | TransT (CVPR21)           | 0.691  | 50 (RTX 2080Ti)  | [Paper](https://arxiv.org/abs/2103.15436)/[Code](https://github.com/chenxin-dlut/TransT) |
     | PrDiMP50 (CVPR20)         | 0.680  | 30 (Unkown GPU)  | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | TrDimp (CVPR21)           | 0.675  | 26 (GTX 1080Ti)  | [Paper](https://arxiv.org/abs/2103.11681)/[Code](https://github.com/594422814/TransformerTrack) |
     | DiMP50 (ICCV19)           | 0.654  | 43 (GTX 1080)    | [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | Siam R-CNN (CVPR20)       | 0.649  | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     
     * **Baseline: SiamRPN++, Ocean, DiMP, PrDiMP, Siam R-CNN**

* **TC-128**

     | Tracker                   |   Success Score  | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | Siam R-CNN (CVPR20)       | 0.649  | 5 (Tesla V100)   | [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) | 
     
     * **Baseline: SiamRPN++, Ocean, DiMP, PrDiMP, Siam R-CNN**
     
 
 * **VOT2020:star2:**  

     | Tracker                   | EAO    | Accuracy (A) | Robustness (R) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | RPT (ECCVW20):star2:      | 0.530 |  XXX | XXX   | [Paper](https://arxiv.org/pdf/2008.03467.pdf)/[Code](https://github.com/zhanght021/RPT) |
     | AlphaRef:star2:           | 0.482 |  XXX | XXX   | [Paper](https://arxiv.org/abs/2007.02024)/[Code](https://github.com/MasterBin-IIAU/AlphaRefine) |     

 * **VOT2018:star2:**  
    
     | Tracker                   | EAO    | Accuracy (A) | Robustness (R) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | RPT (ECCVW20)             | 0.510  |  0.629  |  0.103  | [Paper](https://arxiv.org/pdf/2008.03467.pdf)/[Code](https://github.com/zhanght021/RPT)  |    
     | Ocean (ECCV20)            | 0.489  |  0.592  | 0.117   | [Paper](https://arxiv.org/abs/2006.10721)/[Code](https://github.com/researchmm/TracKit) |  
     | D3S (CVPR20)              | 0.489  |  0.640  | 0.150   | [Paper](https://arxiv.org/pdf/1911.08862.pdf)/[Code](https://github.com/alanlukezic/d3s) |  
     | SiamAttn (CVPR20)         | 0.470  |  0.630  | 0.160   | [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | MAML-Retina (CVPR20)      | 0.452  |  0.604  | 0.159   | [Paper](https://arxiv.org/pdf/2004.00830.pdf)/[Code]() | 
     | SiamBAN (CVPR20)          | 0.452  |  0.597  | 0.178   | [Paper](https://arxiv.org/pdf/2003.06761.pdf)/[Code](https://github.com/hqucv/siamban) |  
     | PG-Net (ECCV20)           | 0.447  |  0.618  | 0.192   | [Paper]()/[Code]()|
     | PrDiMP50 (CVPR20)         | 0.442  |  0.618  | 0.165   | [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |  
     | DiMP50 (ICCV19)           | 0.440  |  0.587  | 0.153   | [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | SiamFC++GoogLeNet (AAAI20)| 0.426  |  0.587  | 0.183   | [Paper](https://arxiv.org/pdf/1911.06188.pdf)/[Code](https://github.com/MegviiDetection/video_analyst) |
     | SiamRPN++ (CVPR19)        | 0.414  |  0.600  | 0.234   | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)/[Code](https://github.com/STVIR/pysot) |     
    
    * VOT2018 and VOT2017 have same sequences. VOT2013 to VOT2016 are small-scale and out-of-date. 
    <!--* LADCF is the best tracker in the original [VOT2018](http://prints.vicos.si/publications/365) report. -->
    * The sequences of VOT2019 and VOT2018 have small differences, and few trackers has been tested on VOT2019. 
    * **Only top-10 published trackers are reported here!**

* **VOT2019**   

     | Tracker                   | EAO    | Accuracy (A) | Robustness (R) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|:----------------:|
     | RPT (ECCVW20)             | 0.417  |  0.623  |  0.186  | [Paper](https://arxiv.org/pdf/2008.03467.pdf)|    
     | DRNet (VOT2019)           | 0.395  |  0.605  |  0.261  | [Code](https://github.com/ShuaiBai623/DRNet)|

    * DRNet is the best tracker in the original [VOT2019](http://prints.vicos.si/publications/375) report. 
    * There is a small difference between VOT2019 and VOT2018. VOT2018 is much more popular. 
    * Few trackers was tested on this dataset and performed better than DRNet. 
    

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
     | | | |  
     | [OTB-2015](http://cvlab.hanyang.ac.kr/tracker_benchmark/)       | 100            |      59K/71/3,872/590       |  No | 
     | [TC-128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html)         | 128     | 55K/71/3,872/429   |  No | 
     | [ALOV300++](http://alov300pp.joomlafree.it/)    | 315            |      8.9K/XXXX/XXXX/284      |  No |
     | [NUS-PRO](https://www.ece.nus.edu.sg/lv/pro/nus_pro.html)        | 365            |      135K/146/5,040/371      |  No |       

     * [OTB-2013/OTB-50](http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html) is a subset of OTB-2015. 


## Conference Tracking Papers: 

* **2021:**
     * Transformer Tracking. CVPR, 2021.  <br /> Xin Chen, Bin Yan, Jiawen Zhu, Dong Wang, Xiaoyun yang, Huchuan Lu. [[Paper](https://arxiv.org/abs/2103.15436)][[Code](https://github.com/chenxin-dlut/TransT)]
     
     * Alpha-Refine: Boosting Tracking Performance by Precise Bounding Box Estimation. CVPR, 2021.  <br />  Bin Yan, Xinyu Zhang, Dong Wang, Huchuan Lu, Xiaoyun Yang.  [[Paper](https://arxiv.org/abs/2012.06815)][[Code](https://github.com/MasterBin-IIAU/AlphaRefine)]
     
     * LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search. CVPR, 2021.  <br />  Bin Yan, Houwen Peng, Kan Wu, Dong Wang, Jianlong Fu, Huchuan Lu.  [[Paper](https://arxiv.org/abs/2104.14545)][[Code](https://github.com/researchmm/LightTrack)] 
     
     * Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking. CVPR, 2021.  <br />  Ning Wang, Wengang Zhou, Jie Wang, Houqiang Li.   [[Paper](https://arxiv.org/abs/2103.11681)][[Code](https://github.com/594422814/TransformerTrack)] 
     
     * Graph Attention Tracking. CVPR, 2021.  <br />  Dongyan Guo, Yanyan Shao, Ying Cui, Zhenhua Wang, Liyan Zhang, Chunhua Shen.   [[Paper](https://arxiv.org/abs/2011.11204)][[Code](https://github.com/ohhhyeahhh/SiamGAT)]
     
     * Towards More Flexible and Accurate Object Tracking with Natural Language: Algorithms and Benchmark. CVPR, 2021.  <br /> Xiao Wang, Xiujun Shu, Zhipeng Zhang, Bo Jiang, Yaowei Wang, Yonghong Tian, Feng Wu.   [[Paper]()][[Project](https://sites.google.com/view/langtrackbenchmark/)] 

     * Siamese Natural Language Tracker: Tracking by Natural Language Descriptions with Siamese Trackers. CVPR, 2021.  <br /> Qi Feng, Vitaly Ablavsky, Qinxun Bai, Stan Sclaroff.   [[Paper](https://arxiv.org/abs/1912.02048v2)][[Project]()] 
     
     *  Distractor-Aware Fast Tracking via Dynamic Convolutions and MOT Philosophy. CVPR, 2021.  <br /> Zikai Zhang, Bineng Zhong, Shengping Zhang, Zhenjun Tang, Xin Liu, Zhaoxiang Zhang.   [[Paper](https://arxiv.org/abs/2104.12041)][[Project](https://github.com/hqucv/dmtrack)]  
     
     *  Learning to Filter: Siamese Relation Network for Robust Tracking. CVPR, 2021.  <br />   Siyuan Cheng, Bineng Zhong, Guorong Li, Xin Liu, Zhenjun Tang, Xianxian Li, Jing Wang. [[Paper](https://arxiv.org/abs/2104.00829)][[Code](https://github.com/hqucv/siamrn)] 

     *  STMTrack: Template-free Visual Tracking with Space-time Memory Networks. CVPR, 2021.  <br /> Zhihong Fu, Qingjie Liu, Zehua Fu, Yunhong Wang. [[Paper](https://arxiv.org/abs/2104.00324)][[Code](https://github.com/fzh0917/STMTrack)]
     
     * IoU Attack: Towards Temporally Coherent Black-Box Adversarial Attack for Visual Object Tracking. CVPR, 2021.  <br /> Shuai Jia, Yibing Song, Chao Ma, Xiaokang Yang. [[Paper](https://arxiv.org/abs/2103.14938)][[Code](https://github.com/VISION-SJTU/IoUattack)] 
     
     * Rotation Equivariant Siamese Networks for Tracking.  CVPR, 2021.  <br /> Deepak K. Gupta, Devanshu Arya, Efstratios Gavves.   [[Paper](https://arxiv.org/abs/2012.13078)][[Code](https://github.com/dkgupta90/re-siamnet)] 

     * Progressive Unsupervised Learning for Visual Object Tracking.  CVPR, 2021.  <br /> Qiangqiang Wu, Jia Wan, Antoni B. Chan.   [[Paper]()][[Code]()] 
     
     * MeanShift++: Extremely Fast Mode-Seeking With Applications to Segmentation and Object Tracking.  CVPR, 2021.  <br /> Jennifer Jang, Heinrich Jiang.   [[Paper]()][[Code]()]  

     * CapsuleRRT: Relationships-Aware Regression Tracking via Capsules.  CVPR, 2021.  <br /> Ding Ma, Xiangqian Wu.   [[Paper]()][[Code]()]  



* **2020:**
     * High-Performance Long-Term Tracking with Meta-Updater. CVPR, 2020.  <br /> Kenan Dai, Yunhua Zhang, Dong Wang, Jianhua Li, Huchuan Lu, Xiaoyun Yang. [[Paper]()][[Code](https://github.com/Daikenan/LTMU)]
     
     * Cooling-Shrinking Attack: Blinding the Tracker with Imperceptible Noises. CVPR, 2020.  <br /> Bin Yan, Dong Wang, Huchuan Lu, Xiaoyun Yang. [[Paper]()][[Code](https://github.com/MasterBin-IIAU/CSA)]
     
     * Siam R-CNN: Visual Tracking by Re-Detection. CVPR, 2020.  <br /> Paul Voigtlaender, Jonathon Luiten, Philip H. S. Torr, Bastian Leibe. [[Paper]()][[Code](https://github.com/VisualComputingInstitute/SiamR-CNN)]
     
     * Probabilistic Regression for Visual Tracking. CVPR, 2020.  <br /> Martin Danelljan, Luc Van Gool, Radu Timofte. [[Paper]()][[Code](https://github.com/visionml/pytracking)]
     
     * D3S - A Discriminative Single Shot Segmentation Tracker. CVPR, 2020.  <br /> Alan Lukezic, Jiri Matas, Matej Kristan.  [[Paper]()][[Code](https://github.com/alanlukezic/d3s)]

     * Tracking by Instance Detection: A Meta-Learning Approach. CVPR, 2020.  <br /> Guangting Wang, Chong Luo, Xiaoyan Sun, Zhiwei Xiong, Wenjun Zeng: Tracking by Instance Detection: A Meta-Learning Approach.  [[Paper]()][[Code]()]
     
     * SiamCAR: Siamese Fully Convolutional Classification and Regression for Visual Tracking. CVPR, 2020.  <br />  Dongyan Guo, Jun Wang, Ying Cui, Zhenhua Wang, Shengyong Chen.  [[Paper]()][[Code](https://github.com/ohhhyeahhh/SiamCAR)]
     
     * Siamese Box Adaptive Network for Visual Tracking. CVPR, 2020.  <br />  Zedu Chen, Bineng Zhong, Guorong Li, Shengping Zhang, Rongrong Ji.  [[Paper]()][[Code](https://github.com/hqucv/siamban)]
     
     * Deformable Siamese Attention Networks for Visual Object Tracking. CVPR, 2020.  <br />  Yuechen Yu, Yilei Xiong, Weilin Huang, Matthew R. Scott.  [[Paper]()][[Code]()] 
     
     * MAST: A Memory-Augmented Self-supervised Tracker. CVPR, 2020.  <br /> Zihang Lai, Erika Lu, Weidi Xie.  [[Paper]()][[Code](https://github.com/zlai0/MAST)] 
     
     * ROAM: Recurrently Optimizing Tracking Model. CVPR, 2020.  <br /> Tianyu Yang, Pengfei Xu, Runbo Hu, Hua Chai, Antoni B. Chan.  [[Paper]()][[Code](https://github.com/skyoung/ROAM)]
     
     * AutoTrack: Towards High-Performance Visual Tracking for UAV with Automatic Spatio-Temporal Regularization. CVPR, 2020.  <br /> 
Yiming Li, Changhong Fu, Fangqiang Ding, Ziyuan Huang, Geng Lu.  [[Paper]()][[Code](https://github.com/vision4robotics/AutoTrack)] 

* **2019:**
     * Unsupervised Deep Tracking. CVPR, 2019.  <br /> Ning Wang, Yibing Song, Chao Ma, Wengang Zhou, Wei Liu, Houqiang Li. [[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Unsupervised_Deep_Tracking_CVPR_2019_paper.pdf)][[Code](https://github.com/594422814/UDT)]

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
     
     * Deep Meta Learning for Real-Time Target-Aware Visual Tracking. ICCV, 2019.  <br /> Janghoon Choi, Junseok Kwon, Kyoung Mu Lee. [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Deep_Meta_Learning_for_Real-Time_Target-Aware_Visual_Tracking_ICCV_2019_paper.pdf)] 
     
     * 'Skimming-Perusal' Tracking: A Framework for Real-Time and Robust Long-Term Tracking. ICCV, 2019.  <br />  Bin Yan, Haojie Zhao, Dong Wang, Huchuan Lu, Xiaoyun Yang. [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Skimming-Perusal_Tracking_A_Framework_for_Real-Time_and_Robust_Long-Term_Tracking_ICCV_2019_paper.pdf)] 
     
     * Learning Aberrance Repressed Correlation Filters for Real-Time UAV Tracking. ICCV, 2019.  <br />  Ziyuan Huang, Changhong Fu, Yiming Li, Fuling Lin, Peng Lu. [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Learning_Aberrance_Repressed_Correlation_Filters_for_Real-Time_UAV_Tracking_ICCV_2019_paper.pdf)] 
     
     * Physical Adversarial Textures That Fool Visual Object Tracking. ICCV, 2019.  <br />  Rey Reza Wiyatno, Anqi Xu. [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wiyatno_Physical_Adversarial_Textures_That_Fool_Visual_Object_Tracking_ICCV_2019_paper.pdf)] 
     
     * GradNet: Gradient-Guided Network for Visual Object Tracking. ICCV, 2019.  <br /> Peixia Li, Boyu Chen, Wanli Ouyang, Dong Wang, Xiaoyun Yang, Huchuan Lu. [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_GradNet_Gradient-Guided_Network_for_Visual_Object_Tracking_ICCV_2019_paper.pdf)] 
     
     * Learning Discriminative Model Prediction for Tracking. ICCV, 2019.  <br />  Goutam Bhat, Martin Danelljan, Luc Van Gool, Radu Timofte. [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Bhat_Learning_Discriminative_Model_Prediction_for_Tracking_ICCV_2019_paper.pdf)] 
     
     * Joint Group Feature Selection and Discriminative Filter Learning for Robust Visual Object Tracking. ICCV, 2019.  <br /> Tianyang Xu, Zhen-Hua Feng, Xiao-Jun Wu, Josef Kittler. [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Joint_Group_Feature_Selection_and_Discriminative_Filter_Learning_for_Robust_ICCV_2019_paper.pdf)] 
     
     * CDTB: A Color and Depth Visual Object Tracking Dataset and Benchmark. ICCV, 2019.  <br />  Alan Lukezic, Ugur Kart, Jani Kapyla, Ahmed Durmush, Joni-Kristian Kamarainen, Jiri Matas, Matej Kristan. [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lukezic_CDTB_A_Color_and_Depth_Visual_Object_Tracking_Dataset_and_ICCV_2019_paper.pdf)]

* **2018:**
     * Context-Aware Deep Feature Compression for High-Speed Visual Tracking. CVPR, 2018.  <br /> Jongwon Choi, Hyung Jin Chang, Tobias Fischer, Sangdoo Yun, Kyuewang Lee, Jiyeoup Jeong, Yiannis Demiris, Jin Young Choi. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_Context-Aware_Deep_Feature_CVPR_2018_paper.pdf)] 
     
     * Correlation Tracking via Joint Discrimination and Reliability Learning. CVPR, 2018.  <br /> Chong Sun, Dong Wang, Huchuan Lu, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Correlation_Tracking_via_CVPR_2018_paper.pdf)]

     * Hyperparameter Optimization for Tracking With Continuous Deep Q-Learning. CVPR, 2018.  <br /> Xingping Dong, Jianbing Shen, Wenguan Wang, Yu Liu, Ling Shao, Fatih Porikli. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Hyperparameter_Optimization_for_CVPR_2018_paper.pdf)] 
     
     * End-to-End Flow Correlation Tracking With Spatial-Temporal Attention. CVPR, 2018.  <br />  Zheng Zhu, Wei Wu, Wei Zou, Junjie Yan. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhu_End-to-End_Flow_Correlation_CVPR_2018_paper.pdf)] 
     
     * Efficient Diverse Ensemble for Discriminative Co-Tracking. CVPR, 2018.  <br />  Kourosh Meshgi, Shigeyuki Oba, Shin Ishii. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Meshgi_Efficient_Diverse_Ensemble_CVPR_2018_paper.pdf)] 
     
     * A Twofold Siamese Network for Real-Time Object Tracking. CVPR, 2018.  <br /> Anfeng He, Chong Luo, Xinmei Tian, Wenjun Zeng. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_A_Twofold_Siamese_CVPR_2018_paper.pdf)] 
     
     * Multi-Cue Correlation Filters for Robust Visual Tracking. CVPR, 2018.  <br /> Ning Wang, Wengang Zhou, Qi Tian, Richang Hong, Meng Wang, Houqiang Li. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Multi-Cue_Correlation_Filters_CVPR_2018_paper.pdf)] 
     
     * Learning Attentions: Residual Attentional Siamese Network for High Performance Online Visual Tracking. CVPR, 2018.  <br />  Qiang Wang, Zhu Teng, Junliang Xing, Jin Gao, Weiming Hu, Stephen Maybank. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_Attentions_Residual_CVPR_2018_paper.pdf)] 
     
     * SINT++: Robust Visual Tracking via Adversarial Positive Instance Generation. CVPR, 2018.  <br />  Xiao Wang, Chenglong Li, Bin Luo, Jin Tang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SINT_Robust_Visual_CVPR_2018_paper.pdf)] 
     
     * High-Speed Tracking With Multi-Kernel Correlation Filters. CVPR, 2018.  <br />  Ming Tang, Bin Yu, Fan Zhang, Jinqiao Wang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tang_High-Speed_Tracking_With_CVPR_2018_paper.pdf)] 
     
     * Learning Spatial-Temporal Regularized Correlation Filters for Visual Tracking. CVPR, 2018.  <br />  Feng Li, Cheng Tian, Wangmeng Zuo, Lei Zhang, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Learning_Spatial-Temporal_Regularized_CVPR_2018_paper.pdf)] 
     
     * Learning Spatial-Aware Regressions for Visual Tracking. CVPR, 2018.  <br /> Chong Sun, Dong Wang, Huchuan Lu, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Learning_Spatial-Aware_Regressions_CVPR_2018_paper.pdf)] 
     
     * High Performance Visual Tracking With Siamese Region Proposal Network. CVPR, 2018.  <br />  Bo Li, Junjie Yan, Wei Wu, Zheng Zhu, Xiaolin Hu. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)]

     * VITAL: VIsual Tracking via Adversarial Learning. CVPR, 2018.  <br /> Yibing Song, Chao Ma, Xiaohe Wu, Lijun Gong, Linchao Bao, Wangmeng Zuo, Chunhua Shen, Rynson W.H. Lau, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Song_VITAL_VIsual_Tracking_CVPR_2018_paper.pdf)]

     * Distractor-aware Siamese Networks for Visual Object Tracking. ECCV, 2018.  <br /> Zheng Zhu, Qiang Wang, Bo Li, Wei Wu, Junjie Yan, Weiming Hu. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zheng_Zhu_Distractor-aware_Siamese_Networks_ECCV_2018_paper.pdf)] 
     
     * Learning Dynamic Memory Networks for Object Tracking. ECCV, 2018.  <br /> Tianyu Yang, Antoni B. Chan. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tianyu_Yang_Learning_Dynamic_Memory_ECCV_2018_paper.pdf)] 
     
     * TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild. ECCV, 2018.  <br />  Matthias Muller, Adel Bibi, Silvio Giancola, Salman Alsubaihi, Bernard Ghanem. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Matthias_Muller_TrackingNet_A_Large-Scale_ECCV_2018_paper.pdf)] 
     
     * Structured Siamese Network for Real-Time Visual Tracking. ECCV, 2018.  <br /> Yunhua Zhang, Lijun Wang, Jinqing Qi, Dong Wang, Mengyang Feng, Huchuan Lu. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yunhua_Zhang_Structured_Siamese_Network_ECCV_2018_paper.pdf)] 
     
     * Triplet Loss in Siamese Network for Object Tracking. ECCV, 2018.  <br /> Xingping Dong, Jianbing Shen. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xingping_Dong_Triplet_Loss_with_ECCV_2018_paper.pdf)]
     
     * Real-time 'Actor-Critic' Tracking. ECCV, 2018.  <br />  Boyu Chen, Dong Wang, Peixia Li, Shuang Wang, Huchuan Lu. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Boyu_Chen_Real-time_Actor-Critic_Tracking_ECCV_2018_paper.pdf)] 
     
     * Joint Representation and Truncated Inference Learning for Correlation Filter based Tracking. ECCV, 2018.  <br />  Yingjie Yao, Xiaohe Wu, Lei Zhang, Shiguang Shan, Wangmeng Zuo. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yingjie_Yao_Joint_Representation_and_ECCV_2018_paper.pdf)] 
     
     * Visual Tracking via Spatially Aligned Correlation Filters Network. ECCV, 2018.  <br />  Mengdan Zhang, Qiang Wang, Junliang Xing, Jin Gao, Peixi Peng, Weiming Hu, Steve Maybank. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/mengdan_zhang_Visual_Tracking_via_ECCV_2018_paper.pdf)] 
     
     * Deep Reinforcement Learning with Iterative Shift for Visual Tracking. ECCV, 2018.  <br />  Liangliang Ren, Xin Yuan, Jiwen Lu, Ming Yang, Jie Zhou. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liangliang_Ren_Deep_Reinforcement_Learning_ECCV_2018_paper.pdf)] 
     
     * Cross-Modal Ranking with Soft Consistency and Noisy Labels for Robust RGB-T Tracking. ECCV, 2018.  <br /> Chenglong Li, Chengli Zhu, Yan Huang, Jin Tang, Liang Wang. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenglong_Li_Cross-Modal_Ranking_with_ECCV_2018_paper.pdf)]
     
     * Long-term Tracking in the Wild: a Benchmark. ECCV, 2018.  <br /> Jack Valmadre, Luca Bertinetto, Joao F. Henriques, Ran Tao, Andrea Vedaldi, Arnold W.M. Smeulders, Philip H.S. Torr, Efstratios Gavves. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Efstratios_Gavves_Long-term_Tracking_in_ECCV_2018_paper.pdf)] 
     
     * Deep Regression Tracking with Shrinkage Loss. ECCV, 2018.  <br /> Xiankai Lu, Chao Ma, Bingbing Ni, Xiaokang Yang, Ian Reid, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.pdf)]

     * Unveiling the Power of Deep Tracking. ECCV, 2018.  <br /> Goutam Bhat, Joakim Johnander, Martin Danelljan, Fahad Shahbaz Khan, Michael Felsberg. [[Paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Goutam_Bhat_Unveiling_the_Power_ECCV_2018_paper.pdf)]

* **2017:**
     * Context-Aware Correlation Filter Tracking. CVPR, 2017.  <br /> Matthias Mueller, Neil Smith, Bernard Ghanem. [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mueller_Context-Aware_Correlation_Filter_CVPR_2017_paper.pdf)]
     
     * Superpixel-Based Tracking-By-Segmentation Using Markov Chains. CVPR, 2017.  <br /> Donghun Yeo, Jeany Son, Bohyung Han, Joon Hee Han. [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeo_Superpixel-Based_Tracking-By-Segmentation_Using_CVPR_2017_paper.pdf)] 
     
     * Action-Decision Networks for Visual Tracking With Deep Reinforcement Learning. CVPR, 2017.  <br /> Sangdoo Yun, Jongwon Choi, Youngjoon Yoo, Kimin Yun, Jin Young Choi. [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yun_Action-Decision_Networks_for_CVPR_2017_paper.pdf)] 
     
     * End-To-End Representation Learning for Correlation Filter Based Tracking. CVPR, 2017.  <br /> Jack Valmadre, Luca Bertinetto, Joao Henriques, Andrea Vedaldi, Philip H. S. Torr. [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Valmadre_End-To-End_Representation_Learning_CVPR_2017_paper.pdf)] 
     
     * Large Margin Object Tracking With Circulant Feature Maps. CVPR, 2017.  <br /> Mengmeng Wang, Yong Liu, Zeyi Huang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Large_Margin_Object_CVPR_2017_paper.pdf)] 
     
     * Multi-Task Correlation Particle Filter for Robust Object Tracking. CVPR, 2017.  <br /> Tianzhu Zhang, Changsheng Xu, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Multi-Task_Correlation_Particle_CVPR_2017_paper.pdf)] 
     
     * Attentional Correlation Filter Network for Adaptive Visual Tracking. CVPR, 2017.  <br /> Jongwon Choi, Hyung Jin Chang, Sangdoo Yun, Tobias Fischer, Yiannis Demiris, Jin Young Choi. [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Choi_Attentional_Correlation_Filter_CVPR_2017_paper.pdf)]  
     
     * Robust Visual Tracking Using Oblique Random Forests. CVPR, 2017.  <br /> Le Zhang, Jagannadan Varadarajan, Ponnuthurai Nagaratnam Suganthan, Narendra Ahuja, Pierre Moulin. [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Robust_Visual_Tracking_CVPR_2017_paper.pdf)] 
     
     * Tracking by Natural Language Specification. CVPR, 2017.  <br /> Zhenyang Li, Ran Tao, Efstratios Gavves, Cees G. M. Snoek, Arnold W.M. Smeulders. [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Tracking_by_Natural_CVPR_2017_paper.pdf)] 
     
     * ECO: Efficient Convolution Operators for Tracking. CVPR, 2017.  <br /> Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, Michael Felsberg. [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Danelljan_ECO_Efficient_Convolution_CVPR_2017_paper.pdf)] 
     
     * Learning Policies for Adaptive Tracking With Deep Feature Cascades. ICCV, 2017.  <br /> Chen Huang, Simon Lucey, Deva Ramanan. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Learning_Policies_for_ICCV_2017_paper.pdf)] 
     
     * Tracking as Online Decision-Making: Learning a Policy From Streaming Videos With Reinforcement Learning. ICCV, 2017.  <br /> James Supancic,III, Deva Ramanan. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Supancic_Tracking_as_Online_ICCV_2017_paper.pdf)] 
     
     * Need for Speed: A Benchmark for Higher Frame Rate Object Tracking. ICCV, 2017.  <br /> Hamed Kiani Galoogahi, Ashton Fagg, Chen Huang, Deva Ramanan, Simon Lucey. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Galoogahi_Need_for_Speed_ICCV_2017_paper.pdf)] 
     
     * Learning Background-Aware Correlation Filters for Visual Tracking. ICCV, 2017.  <br /> Hamed Kiani Galoogahi, Ashton Fagg, Simon Lucey. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Galoogahi_Learning_Background-Aware_Correlation_ICCV_2017_paper.pdf)] 
     
     * Robust Object Tracking Based on Temporal and Spatial Deep Networks. ICCV, 2017.  <br /> Zhu Teng, Junliang Xing, Qiang Wang, Congyan Lang, Songhe Feng, Yi Jin. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Teng_Robust_Object_Tracking_ICCV_2017_paper.pdf)] 
     
     * Learning Dynamic Siamese Network for Visual Object Tracking. ICCV, 2017.  <br /> Qing Guo, Wei Feng, Ce Zhou, Rui Huang, Liang Wan, Song Wang. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Guo_Learning_Dynamic_Siamese_ICCV_2017_paper.pdf)] 
     
     * CREST: Convolutional Residual Learning for Visual Tracking. ICCV, 2017.  <br /> Yibing Song, Chao Ma, Lijun Gong, Jiawei Zhang, Rynson W. H. Lau, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Song_CREST_Convolutional_Residual_ICCV_2017_paper.pdf)] 
     
     * Beyond Standard Benchmarks: Parameterizing Performance Evaluation in Visual Object Tracking. ICCV, 2017.  <br /> Luka Cehovin Zajc, Alan Lukezic, Ales Leonardis, Matej Kristan. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zajc_Beyond_Standard_Benchmarks_ICCV_2017_paper.pdf)] 
     
     * Parallel Tracking and Verifying: A Framework for Real-Time and High Accuracy Visual Tracking. ICCV, 2017.  <br />
Heng Fan, Haibin Ling. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Fan_Parallel_Tracking_and_ICCV_2017_paper.pdf)] 
     
     * Non-Rigid Object Tracking via Deformable Patches Using Shape-Preserved KCF and Level Sets. ICCV, 2017.  <br /> Xin Sun, Ngai-Man Cheung, Hongxun Yao, Yiluan Guo. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Sun_Non-Rigid_Object_Tracking_ICCV_2017_paper.pdf)] 
     
     * Learning Policies for Adaptive Tracking With Deep Feature Cascades. ICCV, 2017.  <br /> Chen Huang, Simon Lucey, Deva Ramanan. [[Paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Learning_Policies_for_ICCV_2017_paper.pdf)] 

* **2016:**
     * Beyond Local Search: Tracking Objects Everywhere With Instance-Specific Proposals. CVPR, 2016.  <br /> Gao Zhu, Fatih Porikli, Hongdong Li. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhu_Beyond_Local_Search_CVPR_2016_paper.pdf)] 
     
     * STCT: Sequentially Training Convolutional Networks for Visual Tracking. CVPR, 2016.  <br /> Lijun Wang, Wanli Ouyang, Xiaogang Wang, Huchuan Lu. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Wang_STCT_Sequentially_Training_CVPR_2016_paper.pdf)] 
     
     * Staple: Complementary Learners for Real-Time Tracking. CVPR, 2016.  <br /> Luca Bertinetto, Jack Valmadre, Stuart Golodetz, Ondrej Miksik, Philip H. S. Torr. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Bertinetto_Staple_Complementary_Learners_CVPR_2016_paper.pdf)] 
     
     * Siamese Instance Search for Tracking. CVPR, 2016.  <br /> Ran Tao, Efstratios Gavves, Arnold W.M. Smeulders. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Tao_Siamese_Instance_Search_CVPR_2016_paper.pdf)] 
     
     * Adaptive Decontamination of the Training Set: A Unified Formulation for Discriminative Visual Tracking. CVPR, 2016.  <br /> Martin Danelljan, Gustav Hager, Fahad Shahbaz Khan, Michael Felsberg. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Danelljan_Adaptive_Decontamination_of_CVPR_2016_paper.pdf)] 
     
     * Adaptive Decontamination of the Training Set: A Unified Formulation for Discriminative Visual Tracking. CVPR, 2016.  <br /> Martin Danelljan, Gustav Hager, Fahad Shahbaz Khan, Michael Felsberg. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Danelljan_Adaptive_Decontamination_of_CVPR_2016_paper.pdf)] 
     
     * Recurrently Target-Attending Tracking. CVPR, 2016.  <br /> Zhen Cui, Shengtao Xiao, Jiashi Feng, Shuicheng Yan. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Cui_Recurrently_Target-Attending_Tracking_CVPR_2016_paper.pdf)] 
     
     * In Defense of Sparse Tracking: Circulant Sparse Tracker. CVPR, 2016.  <br /> Tianzhu Zhang, Adel Bibi, Bernard Ghanem. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_In_Defense_of_CVPR_2016_paper.pdf)] 

     * Object Tracking via Dual Linear Structured SVM and Explicit Feature Map. CVPR, 2016.  <br /> Jifeng Ning, Jimei Yang, Shaojie Jiang, Lei Zhang, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Ning_Object_Tracking_via_CVPR_2016_paper.pdf)] 
     
     * Learning Multi-Domain Convolutional Neural Networks for Visual Tracking. CVPR, 2016.  <br /> Hyeonseob Nam, Bohyung Han. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Nam_Learning_Multi-Domain_Convolutional_CVPR_2016_paper.pdf)] 
     
     * Hedged Deep Tracking. CVPR, 2016.  <br /> Yuankai Qi, Shengping Zhang, Lei Qin, Hongxun Yao, Qingming Huang, Jongwoo Lim, Ming-Hsuan Yang. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Qi_Hedged_Deep_Tracking_CVPR_2016_paper.pdf)] 
     
     * Target Response Adaptation for Correlation Filter Tracking. ECCV, 2016.  <br />  Adel Bibi, Matthias Mueller, Bernard Ghanem. 
     
     * Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking. ECCV, 2016.  <br />  Martin Danelljan, Andreas Robinson, Fahad Khan, Michael Felsberg.
     
     * Structural Correlation Filter for Robust Visual Tracking. CVPR, 2016.  <br /> Si Liu, Tianzhu Zhang, Xiaochun Cao, Changsheng Xu. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Structural_Correlation_Filter_CVPR_2016_paper.pdf)] 
     
     * Visual Tracking Using Attention-Modulated Disintegration and Integration. CVPR, 2016.  <br /> Jongwon Choi, Hyung Jin Chang, Jiyeoup Jeong, Yiannis Demiris, Jin Young Choi. [[Paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Choi_Visual_Tracking_Using_CVPR_2016_paper.pdf)] 
     
     * A Benchmark and Simulator for UAV Tracking. ECCV, 2016.  <br /> Matthias Mueller, Bernard Ghanem, Neil Smith. 
     
     * Distractor-supported single target tracking in extremely cluttered scenes. ECCV, 2016.  <br /> Jingjing Xiao, Linbo Qiao, Rustam Stolkin, leš Leonardis. 
     
     * Real-Time Visual Tracking: Promoting the Robustness of Correlation Filter Learning. ECCV, 2016.  <br /> Yao Sui, Ziming Zhang, Guanghui Wang, Yafei Tang, Li Zhang. 


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
    
    * Visual Tracking by Sampling Tree-Structured Graphical Models. ECCV, 2014.  <br /> Seunghoon Hong, Bohyung Han. 
    
    * Description-Discrimination Collaborative Tracking. ECCV, 2014.  <br /> Dapeng Chen, Zejian Yuan,Gang Hua, Yang Wu, Nanning Zheng. 
    
    * Online, Real-Time Tracking using a Category-to-Individual Detector. ECCV, 2014.  <br /> David Hall, Pietro Perona. 
     
     * Robust Visual Tracking with Double Bounding Box Model. ECCV, 2014.  <br /> Junseok Kwon, Junha Roh, Kyoung Mu Lee, Luc Van Gool. 
     
     * Transfer Learning Based Visual Tracking with Gaussian Process Regression. ECCV, 2014.  <br /> Jin Gao, Haibin Ling, Weiming Hu, Junliang Xing. 
     
     * Online Graph-Based Tracking. ECCV, 2014.  <br /> Hyeonseob Nam , Seunghoon Hong, Bohyung Han. 
     
     * Fast Visual Tracking via Dense Spatio-Temporal Context Learning. ECCV, 2014.  <br /> Kaihua Zhang, Lei Zhang, Qingshan Liu, David Zhang, Ming-Hsuan Yang. 
     
     * Extended Lucas-Kanade Tracking. ECCV, 2014.  <br /> Shaul Oron, Aharon Bar-Hillel, Shai Avidan. 
     
     * Appearances can be deceiving: Learning visual tracking from few trajectory annotations. ECCV, 2014.  <br /> Santiago Manen, Junseok Kwon, Matthieu Guillaumin, Luc Van Gool. 
     
     * Tracking using Multilevel Quantizations. ECCV, 2014.  <br /> Zhibin Hong, Chaohui Wang, Xue Mei, Danil Prokhorov, Dacheng Tao. 
     
     * Occlusing and Motion Reasoning for Long-term Tracking. ECCV, 2014.  <br /> Yang Hua, Karteek Alahari, Cordelia Schmid. 
     
     * MEEM: Robust Tracking via Multiple Experts using Entropy Minimization. ECCV, 2014.  <br /> Jianming Zhang, Shugao Ma, Stan Sclaroff. 
     
     * A Superior Tracking Approach: Building a strong Tracker through Fusion. ECCV, 2014.  <br /> Christian Bailer, Alain Pagani, Didier Stricker. 

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
     
* **Before 2013:**
     




* **Survey & Book:**


    * Handcrafted and Deep Trackers: Recent Visual Object Tracking Approaches and Trends. ACM CS, 2019. <br />  Mustansar Fiaz, Arif Mahmood, Sajid Javed, Soon Ki Jung.
     
    * Online Visual Tracking. Springer, 2019. <br />  Huchuan Lu, Dong Wang.

    * Deep Visual Tracking: Review and Experimental Comparison. PR, 2018. <br />  Peixia Li, Dong Wang, Lijun Wang, Huchuan Lu.
    
    * Visual Tracking: An Experimental Survey. IEEE TPAMI, 2014. <br /> Arnold W. M. Smeulders, Dung Manh Chu, Rita Cucchiara, Simone Calderara, Afshin Dehghan, Mubarak Shah. 
    
    * A Survey of Appearance Models in Visual Object Tracking. ACM TIST, 2013. <br /> Xi Li, Weiming Hu, Chunhua Shen, Zhongfei Zhang, Anthony R. Dick, Anton van den Hengel.  
    
    * Object Tracking: A Survey. ACM CS, 2006. <br /> Alper Yilmaz, Omar Javed, Mubarak Shah. 

* **Resources:**

     * [TracKit[Microsoft]](https://github.com/researchmm/TracKit)：https://github.com/researchmm/TracKit
     
     * [pytracking[visionml]](https://github.com/visionml/pytracking)：https://github.com/visionml/pytracking
     
     * [pysot[STVIR]](https://github.com/STVIR/pysot)：https://github.com/STVIR/pysot
     
     * [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit)：https://github.com/StrangerZhang/pysot-toolkit
     
     * [SOT-MOT-Comparision](https://github.com/JudasDie/Comparision)：https://github.com/JudasDie/Comparision
     
     * [SiamTrackers](https://github.com/HonglinChu/SiamTrackers)：https://github.com/HonglinChu/SiamTrackers
     
     * [CFTrackers](https://github.com/HonglinChu/CFTrackers)：https://github.com/HonglinChu/CFTrackers
     
     * [Single-Object-Tracking-Paper-List](https://github.com/wangxiao5791509/Single_Object_Tracking_Paper_List)：https://github.com/wangxiao5791509/Single_Object_Tracking_Paper_List

     * [Benchmark-Results](https://github.com/foolwood/benchmark_results)：https://github.com/foolwood/benchmark_results
     
* **Online Reports[Chinese]:**

     * [基于计算机视觉的目标跟踪算法概览(早期算法)](https://www.bilibili.com/video/BV1ex411m7Ty): https://www.bilibili.com/video/BV1ex411m7Ty
     * [从零开始的单目标跟踪算法世界(pytracking为主)](https://www.bilibili.com/video/BV1WK4y1C7JG)：https://www.bilibili.com/video/BV1WK4y1C7JG
