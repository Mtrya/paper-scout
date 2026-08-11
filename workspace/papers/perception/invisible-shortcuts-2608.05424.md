# Invisible Shortcuts: Why Vision Encoders Know Your Camera

Vladan Stojnić<sup>1</sup>, Ryan Ramos<sup>2</sup>, Giorgos Kordopatis-Zilos<sup>1</sup>, Noa Garcia<sup>2</sup>, and Giorgos Tolias<sup>1</sup>

<sup>1</sup> VRG, FEE, Czech Technical University in Prague 2 The University of Osaka

![](images/c61ae1c6fb4683f881a66534d23f46dc9801c43a11bc6559ee069ba476f5ebaa.jpg)

![](images/d5a6bb0374b59e61feb8242bf90af22d8997ca09c0a8454a2ece031fa686686e.jpg)

![](images/760632520eed6b3005ec2fe5bf9680359d0ffc20a388d3fa83c4f6f8d246759c.jpg)

![](images/ff94b2b4045956463040b6f0a201a7ca05460c05793a7020af8ae84f0a871a09.jpg)  
Fig. 1: Metadata distribution per semantic category. Diferent classes (ImageNet21k) and topics (Re-LAION-2B) have distinct distributions, indicating correlations between metadata and semantics. These correlations create shortcut signals that models exploit during large-scale pretraining. Global: overall dataset distribution.

Abstract. Deep vision models exploit shortcuts, relying on cues that correlate with supervision signals. Prior work has focused on visible biases, such as object-background or texture correlations. We identify a diferent source of shortcut learning: invisible metadata traces embedded at the pixel level, for metadata such as image processing and photo acquisition. We hypothesize that large-scale semantic supervision, whether through categorical labels (ImageNet) or billion-scale captions (LAION), naturally induces metadata-semantics correlations during pretraining, leading models to convert low-level signals into predictive features. By introducing controlled metadata-semantics correlations, we show that stronger ones produce systematically higher sensitivity to metadata traces and larger performance degradation under metadata distribution shifts. We further explore mitigation strategies applied during and after pretraining that reduce sensitivity not only to targeted metadata but also to unseen ones, without sacrificing performance on downstream tasks. Metadata sensitivity also has a positive side: it partly explains the strong generated-image detection ability of some encoders, while its mitigation can improve out-of-distribution generalization. Code: https://github.com/ryan-caesar-ramos/visual-encoder-traces

## 1 Introduction

Deep neural networks are known to rely on shortcuts [4,14,62,66,72,81]. When a predictive cue is present in training data, models exploit it regardless of whether it aligns with the intended task, since the objective is simply to minimize error.

Whether this is harmful depends on the stability of these correlations; when they shift, performance degrades. This phenomenon, studied under shortcut learning [14] and spurious correlations [32, 56], has been extensively analyzed in vision, including object–background dependencies [21,45,64], color–label correlations [41], texture versus shape bias [15], watermark artifacts [31], and synthetic benchmarks [2, 56]. These cues are visible and interpretable to humans.

We study shortcuts largely invisible to the human eye, namely metadata traces, including processing parameters such as JPEG compression and sharpening, and acquisition parameters such as focal length and camera model. Although imperceptible at a semantic level, they leave low-level signals.

Recent work shows that distribution shifts in image metadata afect the predictions of pretrained vision models [52], revealing that representations encode metadata traces, but the origin remains unclear. We argue that metadata sensitivity arises from metadata–semantics correlations in pretraining datasets: images from certain semantic categories are often captured or processed under similar conditions, creating systematic associations between labels or captions and metadata attributes such as camera parameters or JPEG quality (Fig. 1). Models trained with semantic supervision exploit these correlations as shortcuts, relying on metadata-derived pixel signals rather than semantic content, as such signals are simpler than high-level concepts and optimization favors them [62].

We test this hypothesis across both regimes of large-scale semantic supervision: categorical label supervision on ImageNet and caption supervision on LAION, measuring the metadata-semantics correlations in each and introducing controlled ones during training. As correlation strength increases, models become progressively more sensitive to metadata. Motivated by this, we investigate mitigation both during and after pretraining. Interestingly, using a single attribute, e.g. JPEG compression, as the correlation source or mitigation target also changes sensitivity to other metadata, e.g. resizing or camera model, suggesting that models exploit a broad class of low-level traces rather than attributespecific artifacts. Notably, this contradicts prior work [31], where mitigating one type of bias amplified others.

Our main contribution is to show that metadata sensitivity is a result of shortcut learning arising from metadata-semantics correlations in training data, manifesting at scale across both supervised ImageNet and caption-guided LAION pretraining, rather than being specific to a single supervision paradigm. We further show it can be mitigated, both during and after pretraining, with mitigation targeting one attribute generalizing to unseen ones. Finally, such sensitivity is not only a disadvantage: since it reflects a model’s tendency to capture pixel-level signal, which is key to separating generated from real images, it partly explains the strong generated-image detection ability of some encoders [46]. Meanwhile its mitigation can improve out-of-distribution generalization.

## 2 Diagnostic metrics

We use metrics that quantify (i) the extent to which frozen model representations encode metadata and (ii) the degree to which such encoding interferes with semantic prediction ability. We follow Ramos et al. [52], see original work for details. We consider two metadata categories: (i) image processing attributes: JPEG compression parameters (quality, chroma subsampling), sharpening (strength), resizing (factor), and interpolation (algorithm type), and (ii) acquisition attributes: camera make/model, exposure time, aperture, ISO, and focal length.

Metadata Prediction (MP). We measure the strength of metadata encoding in frozen model representations by training a linear classifier to predict metadata labels. For processing metadata, we use ImageNet1k (IN1k) [55], randomly assign processing labels to training and test images, and process them accordingly. For acquisition metadata, we use FlickrExif [52] with predefined train/test splits per acquisition attribute. Accuracy is denoted by ${ \mathrm { M P } } _ { \mathrm { p } }$ and $\mathrm { M P a }$ for processing and acquisition metadata, respectively. Accuracy substantially above chance indicates that the representation contains linearly separable metadata information, while accuracy close to chance suggests invariance to metadata labels.

Semantic Prediction (SP) ability is evaluated by the accuracy of a k-NN classifier predicting semantic labels with original IN1k train/val sets.

Semantic Prediction Distraction – processing $\bf ( S P D _ { p } )$ . To quantify the impact of processing-related metadata on semantic predictions, we use IN1k with a k-NN classifier for semantic labels, where metadata labels of training/exemplar images are modified per test image guided by semantic labels. Images are processed according to the assigned metadata label. Two evaluation setups: (i) Possame: the test image and its semantic positives are assigned to the same processing label, while negatives are assigned to a diferent one. Accuracy in this setup is denoted by $A _ { \mathrm { p o s - s a m e } } .$ (ii) Neg-same: the test image and its semantic negatives are assigned to the same processing label, while positives are assigned to a diferent one. Accuracy in this setup is denoted by $A _ { \mathrm { n e g - s a m e } }$ . We evaluate distraction with $\varDelta _ { p } = | A _ { \mathrm { p o s - s a m e } } - A _ { \mathrm { n e g - s a m e } } |$ quantifying the extent to which processing metadata impacts semantic recognition. Lower values indicate stronger invariance to processing metadata, while higher values indicate stronger sensitivity.

Semantic Prediction Distraction – acquisition $\left( \mathbf { S P D _ { a } } \right)$ . To measure how acquisition metadata interferes with semantic similarity, we use the PairCams [52] dataset. Given a query image from one camera class $e . g$ . smartphones, the task is to retrieve a semantically positive image captured with a diferent camera class $e . g$ . digital cameras. Using recall@1, denoted by $R ,$ we evaluate two cases where semantically negative images vary: (i) same: negatives originate from the same camera class as the query; performance denoted by $R _ { \mathrm { s a m e } }$ , and (ii) diferent: negatives originate from a diferent camera class than the query; performance denoted by $R _ { \mathrm { d i f f } }$ . We define the distraction score as $\varDelta _ { a } = | R _ { \mathrm { s a m e } } - R _ { \mathrm { d i f f } } | .$ , quantifying the extent to which acquisition metadata impacts semantic recognition. Overall. MP quantifies metadata encoding, SPD quantifies metadata interference with semantic recognition, and SP captures whether training data correlations or mitigation strategies degrade semantic prediction performance.

## 3 Why are vision encoders sensitive to image metadata

The work of Ramos et al. [52] shows that vision encoders are sensitive to image metadata, and that the presence of metadata traces in features of pretrained models has consequences on downstream performance. However, the causes of these traces remain unclear. We hypothesize that they stem from correlations between metadata and semantic labels in pretraining datasets: such correlations let metadata labels and their pixel-level signal act as a shortcut for predicting semantic labels, leading the model to encode metadata information. We corroborate our hypothesis in several steps. We first show that these correlations, with both processing and acquisition metadata, exist across standard pretraining datasets, spanning categorical (ImageNet) and caption-based (LAION) supervision. We then use controlled experiments to verify that an encoder’s metadata sensitivity grows as a direct function of correlation strength.

## 3.1 Correlations of semantics and processing metadata in ImageNet

Measuring existing correlations. We focus on JPEG compression parameters, specifically the quality and chroma subsampling, as processing metadata. Each combination of values for these two parameters defines a discrete categorical label for each image. We analyze the association between these metadata and semantic labels in ImageNet1k (IN1k) and ImageNet21k (IN21k) using Cramér’s V [7] measure of association between categorical variables. The values are 0.047 and 0.067 in IN1k and IN21k, respectively. Even though the value itself is low and not easily interpretable, we focus on the relative comparison between the two datasets; correlations are stronger in IN21k than in IN1k.

Impact of correlations. We use models<sup>3</sup> pretrained on IN1k and IN21k to evaluate the impact of correlation strength. In Fig. 2, we present all the diagnostic metrics, SP, MP, and SPD, for diferent backbones. We observe that both types of models have a good ability to predict metadata labels (MP), both processing and acquisition ones, at rates substantially above random chance. While there exist minor diferences in training recipes, the models trained on IN21k consistently demonstrate stronger predictive ability. This indicates that features trained on IN21k encode more metadata information than those trained on IN1k. The SPD measurements further show that metadata afects the semantic prediction abilities of both models, but more so for IN21k. Together, these results show that IN21k features not only encode more metadata information but also let it interfere more strongly with semantic prediction. Overall, these results suggest that the stronger correlations between metadata and semantic labels in IN21k, relative to IN1k, have a greater impact on both the models’ ability to encode metadata information and the sensitivity of their semantic predictions to metadata.

![](images/157afe705957dd86692b88860f17236a37c0ccd31ca94bb3d37ed1d87a1d80f2.jpg)  
Fig. 2: IN1k vs. IN21k pretraining: which training set causes higher metadata sensitivity? Reporting Metadata Prediction (MP), Semantic Prediction (SP), and Semantic Prediction Distraction (SPD) for ConvNeXt [36], FlexiViT [5], Swin [34], and ViT [10] pretrained on IN1k and IN21k. The amount of correlation between JPEG compression parameters and semantic labels is measured to be 0.047 and 0.067 in IN1k and IN21k, respectively, according to Cramér’s V measure. The stronger correlations in IN21k result in consistently more sensitive models.

## 3.2 Controlling correlations of semantics and processing metadata

To further verify that the presence and extent of metadata-semantics correlations impacts model’s sensitivity to metadata, we synthetically introduce correlations into the IN1k training set, train models on these correlated datasets, and evaluate all diagnostic metrics.

Creating stronger correlations. We define 22 distinct metadata labels by combining 11 JPEG quality levels with 2 chroma subsampling modes. All images are then recompressed according to one of these labels, creating a correlation between semantic labels and metadata labels. We randomly assign one specific JPEG compression, i.e. metadata label, to each semantic class. We introduce a probability $p _ { i }$ that determines whether an image is processed using the metadata label assigned to its semantic class or using a randomly sampled metadata label. When $p _ { i } = 1 0 0$ , correlation is maximal, while when $p _ { i } = 0 ,$ metadata is independent of semantics. Note that Cramér’s V is roughly equal to the value of p<sub>i</sub>, e.g. 0.1 and 0.8 for $p _ { i } = 1 0$ and $p _ { i } = 8 0$ , respectively.

Impact of correlations. We train ResNet50 $[ 1 8 ] ^ { 4 }$ on these versions of the IN1k training set created using the mechanism described above, and evaluate the models using the diagnostic metrics. Fig. 3 presents the results, where the larger values of $p _ { i }$ mean stronger correlation between metadata and semantics.

<sup>4</sup> ViT-S/16 [10], and ViT-B/16 [10] results are presented in the supplementary.

![](images/fff2ad1081ccd86cea96b72c865471acabb7b4c61902aa53b9a8f05234edd903.jpg)  
Fig. 3: Impact of correlations between semantics and processing metadata on the model’s sensitivity to metadata. ResNet50 trained on diferent versions of IN1k with manipulated JPEG labels. The larger the $p _ { i } ,$ the stronger the correlation between JPEG labels and semantic labels. None: model trained on original IN1k. Diference between $p _ { i } = 0$ and none: the former has no JPEG-semantics correlations, while the latter has the original IN1k correlations. SP: two types of test sets are used, the original one (none, green color) which is fixed across values of $p _ { i }$ , and $7$ diferent versions (purple color) created by manipulating the JPEG labels of the original test set in the same way as the corresponding training set for each value of $p _ { i }$

We observe that accuracy of predicting JPEG metadata labels (MP) increases noticeably for stronger correlations. For zero correlations it is close to random chance, and lower than for the model trained on the original IN1k, verifying that the original IN1k has some correlations, though weaker than the ones we introduce with $p _ { i } \geq 2 0$ . Since the model is trained on a set where JPEG compression drives the correlation, its ability to predict JPEG labels is expected. More notably, the ability to predict other processing and acquisition attributes (resizing, make, aperture) also increases with the JPEG-semantics correlation, suggesting that the model learns to encode pixel-level signal carrying such metadata.

The semantic prediction ability (SP) is afected by the correlation strength (green bars of bottom-left plot). There is no noticeable drop for weak correlations compared to zero correlations, but as the correlation becomes stronger, the drop becomes more significant. This suggests that strong correlations make the model capture pixel-level signal related to metadata labels rather than complex semantic-level features. Performance is additionally reported for testing on 7 modified versions (purple bars of bottom-left plot) of the IN1k validation, each processed using the same metadata-to-semantics associations as the training set for the corresponding value of $p _ { i }$ . Observe that when evaluating on these transformed test sets, the accuracy increases for increased correlations; this is because the test set is transformed identically to the training set. The model uses the same pixel-level signal that it exploited during training, which is not the case when testing on the original set.

Finally, we observe that the distraction of semantic prediction abilities by metadata (SPD) also increases with the increase in correlation, which suggests that the model relies more on the metadata information for semantic prediction when the correlation is stronger, and thus the interference of metadata information with semantic prediction becomes stronger as well.

## 3.3 Correlations of semantics and acquisition metadata in LAION

We additionally investigate whether correlations exist between semantics and acquisition metadata. The subset of Re-LAION-2B [28] carrying Exif tags, about 40M images, provides an image set annotated with acquisition metadata labels. Since Re-LAION-2B contains captions rather than categorical class labels, we apply topic modeling [17] to cluster caption embeddings and assign each caption to one of the resulting 6, 238 topics, which serves as a discrete semantic label per image. Fig. 1 illustrates examples of such correlations, where for a given acquisition attribute, we plot the frequency distribution of its metadata labels within Re-LAION-2B for two topics with clearly distinct distributions. For instance, images whose captions fall under the topic of football tend to have longer focal lengths than those under plots of land. Similar distinctions arise between weddings and eyewear for aperture, and between birds and kitchens for exposure time. We presume that such correlations can act as a source of shortcut learning, analogous to the processing metadata case, which we verify in the following.

## 3.4 Controlling correlations of semantics and acquisition metadata

Creating stronger and weaker correlations. We create three Re-LAION-2B subsets of equal size, i.e. 6.4M images, with varying amounts of correlation between semantic topics and acquisition metadata labels. Metadata labels are defined as tuples comprising the labels for the following attributes: camera manufacturer, aperture, exposure, and ISO speed. (i) baseline: we sample uniformly, reflecting the natural correlations present in Re-LAION-2B. (ii) stronger: for each metadata label, we select images belonging to its 40 most frequent topics, concentrating each label on a few topics and thus strengthening the topic– metadata association. (iii) weaker: for each metadata label, we identify its least frequent topic and downsample all other topics to that size, equalizing the topic distribution within each metadata label and thus weakening the association.

Impact of correlations. We train ResNet50 models<sup>5</sup> using CLIP loss [51], with results in Fig. 4. Training on the subset with stronger correlations consistently leads to higher SPD, while training on the weaker one tends to yield lower SPD, indicating that acquisition metadata correlations are also exploited as shortcuts. Furthermore, consistently with Sec. 3.2, models trained on the weaker subset exhibit the strongest semantic prediction ability, and those trained on the stronger subset the weakest. The MP results are less conclusive; nevertheless, interference with semantic predictions reflected by SPD is the key signal of metadata sensitivity.

<sup>5</sup> ViT-S/16 [10] and ConvNeXt-T [36] results are presented in the supplementary.

![](images/bf3636abb2785bbb60c135f2007c76eeca2db9a3efa64ac3817ede9a9c1fe5d9.jpg)  
Fig. 4: Impact of correlations between semantics and acquisition metadata on the model’s sensitivity to metadata. ResNet50 trained from scratch using CLIP loss [51] on three Re-LAION-2B subsets with Cramér’s V correlations of 0.396, 0.255, and 0.166, referred to as stronger, baseline, and weaker, respectively.

## 4 Analysis of mitigation approaches

We investigate mitigation in two settings: during training on a standard benchmark (IN1k), and as post-processing of released, widely used foundational models that already exhibit metadata sensitivity. Unlike Sec. 3, these experiments do not rely on artificially introduced correlations, but instead target sensitivity reduction under naturally occurring ones.

## 4.1 Mitigation during training with augmentations

Augmentations during training. Ramos et al. [52] argue that strong data augmentations during training reduce metadata sensitivity. We explore this by training and testing ResNet50 on original IN1k. The default training recipe includes RandomAugment (RA) [8], while we remove some or add additional augmentations to test their impact.

Impact of augmentations. Fig. 5 presents the results of this experiment. Removing any of the RA augmentations commonly increases the model’s ability to capture metadata and its sensitivity. Adding Gaussian blur noticeably reduces sensitivity to processing metadata, but has little impact on acquisition metadata; blurring is thus relatively efective at removing the pixel-level signal that metadata leaves, while harming semantic prediction (SP) only slightly. Most notably, the combination of color jittering, blurring, and grayscale conversion consistently reduces sensitivity by a large factor for both metadata types. This is the set of augmentations used by DINOv2 [47], which is reported to be among the least sensitive foundational models [52].

![](images/a536d77535747d6c93beb5ea403fcf5b51e653f19d2fd5ef973059a20266c4af.jpg)  
Fig. 5: Mitigation during training with augmentations – impact on model sensitivity to metadata. ResNet50 trained on IN1k with a varying set of augmentations. The default case is Random Augment (RA) [8], while we remove or add augmentations from/to the default. all: color jitter, grayscale, and blur.

## 4.2 Post-hoc adversarial mitigation

Adversarial mitigation. Given a pretrained model that maps an image to a D-dimensional representation space, we train a linear layer $f : \mathbb { R } ^ { D }  \mathbb { R } ^ { D }$ 2 as a mitigation component, to transform the representation space and reduce metadata sensitivity. A linear classifier $h : \mathbb { R } ^ { D } \to \bar { \mathbb { R } } ^ { M }$ , applied to the transformed space, predicts among M metadata labels. Training is adversarial [1]: h tries to predict metadata labels, while f tries to make the predictions of h as bad as possible. It proceeds in two steps. (i) Train the metadata prediction model h while keeping the mitigation component frozen, via cross-entropy loss and supervision from metadata labels. (ii) Train the mitigation component f while keeping the metadata prediction model frozen, using three loss terms: one that keeps the transformed feature $f ( x )$ close to the pretrained feature by maximizing their dot product $f ( x ) ^ { \top } x ;$ one that pushes apart features $f ( x )$ and $f ( y )$ of images sharing the same metadata label by minimizing their dot product $f ( x ) ^ { \top } f ( y )$ ; and one that maximizes the entropy of the metadata prediction $h ( f ( x ) )$ for each feature x. These two steps are repeated L times. The process requires access to metadata labels, which we obtain by reprocessing IN1k with diferent JPEG compression parameters; can be used with other attributes similarly.

Impact of adversarial mitigation. Fig. 6 presents the results for this mitigation strategy for diferent contrastive vision language (CLIP [51], ConvNeXt [36], SigLIP [83]), supervised (ViT [10], ConvNeXt [36]), and self-supervised (DINO [6], DINOv2 [47]) models. We observe that such process consistently reduces the sensitivity for a variety of models, both in terms of metadata prediction and also in terms of semantic prediction distraction. The most interesting fact is that the mitigation training is guided by the JPEG compression parameters, but it also reduces the sensitivity to other types of metadata. Regarding the overall semantic prediction ability, it is either maintained or even slightly improved.

![](images/f1b29106fff461b62d255001760dc99a1fdce9ce0d2ab22deccb5dee3c4ccd02.jpg)  
Fig. 6: Post-hoc adversarial mitigation with a linear layer – impact on model sensitivity to metadata. Performance is reported for the original frozen foundational models and for the models adjusted by an additional linear layer trained to mitigate the metadata sensitivity guided by manipulated JPEG labels on IN1k.

## 5 Further analysis

Do Stable Difusion-generated images carry metadata traces? We hypothesize that images generated by models such as Stable Difusion [54] may not contain pixel-level metadata traces, since they do not arise from a physical image formation process and the post-processing pipeline can be kept uniform after generation. To test this, we train models on synthetically generated images for IN1k classes [12, 58] compressed with fixed high-quality JPEG parameters. The resulting models still encode and rely on metadata, with $\mathrm { M P } _ { p } \mathrm { ~ - ~ } \mathrm { J P E G }$ 29.91, MP<sub>p</sub> – Resizing: 61.58, SPD<sub>p</sub> – JPEG: 2.00, and $\mathrm { S P D } _ { p } \gets ]$ Resizing: 6.06, comparable to training on the original ImageNet (none) in Fig. 3. This suggests generative models may implicitly reproduce pixel-level traces correlated with the semantics in their training data.

Metadata sensitive models are good detectors of generated images. Recent work [46,61,65] shows that frozen CLIP features are efective at detecting generated images, which we hypothesize is due to their strong metadata encoding ability. To test this, we apply a k-NN classifier $( k = 1 )$ to detect generated images, as in Ojha et al. [46], using features from the models in Sec. 3.2 with varying metadata sensitivity. Tab. 1 shows that more sensitive models, trained on strong JPEG-semantics correlations, i.e. $p _ { i } = 5 0 \mathrm { o r } 1 0 0$ , detect generated images noticeably better than the one trained on the original IN1k. This validates our hypothesis: metadata-semantics correlations encourage the model to focus on low-level details, a beneficial trait for synthetic image detection.

<table><tr><td rowspan="3">Model</td><td colspan="6">Generative Adversarial Networks</td><td rowspan="2">fakes</td><td rowspan="2">Deep Low level vision Perceptual loss</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Guided</td><td rowspan="2"></td><td rowspan="2">LDM</td><td rowspan="2"></td><td colspan="3">Glide</td><td rowspan="2">DALL-E Avg</td><td rowspan="2"></td></tr><tr><td>Pro- Cycle-</td><td></td><td>Big- Style- Gau- Star-</td><td></td><td>GAN GAN</td><td>CRN IMLE</td><td>200 200</td><td>100</td><td>100 50</td></tr><tr><td></td><td>GAN</td><td>GAN</td><td>GAN</td><td>GAN</td><td></td><td></td><td></td><td></td><td></td><td></td><td>57.3</td><td>59.6</td><td></td><td>steps w/</td><td>/ CFG steps</td><td></td><td>27</td><td>27</td><td>10</td><td></td></tr><tr><td>pi = none</td><td>69.1</td><td>65.3</td><td>49.5</td><td>53.5</td><td>58.7</td><td>55.4</td><td>51.1</td><td></td><td>55.0 54.7</td><td>49.8 52.3</td><td>64.7 68.2</td><td>51.2</td><td>62.2</td><td>53.4</td><td></td><td>62.6 58.3</td><td>60.7</td><td>58.7</td><td>57.4</td><td>57.3</td></tr><tr><td> $p _ { i } = 5 0 \%$  pi = 100% 77.2</td><td>71.4</td><td>66.4 67.0</td><td>49.0 48.4</td><td>58.6 58.4</td><td>58.1 51.4</td><td>60.1 76.0</td><td>56.1 58.7</td><td></td><td>58.3</td><td>53.9</td><td>61.3 61.9</td><td>49.9 55.0</td><td>61.3 62.9</td><td>53.7 56.2</td><td>61.3 60.3</td><td>59.5 60.2 62.5 61.1</td><td>60.4</td><td>57.9</td><td>57.4 57.5</td><td>59.0 60.4</td></tr></table>

Table 1: Metadata sensitive models are good detectors of generated images. Used models are ResNet50 trained on diferent versions of IN1k with manipulated JPEG labels from Sec. 3.2 (original IN1k/none, $p _ { i } = 5 0$ and $p _ { i } = 1 0 0$ with low, strong and maximal correlations, respectively). Binary classification accuracy reported.

![](images/0f613bdc619043026792bdba5a2421928a8b36ea154dcc51a71db1536f24148e.jpg)  
Fig. 7: Ability of internal model layers to predict metadata. Metadata prediction (MP) accuracy across blocks of ResNet50 trained on: (i) original IN1k, (ii) original IN1k using extra augmentations (color jitter, grayscale, and blur augmentations), and (iii) modified IN1k with strong JPEG-semantics correlations. In all cases Random Augment is used.

Layer-wise sensitivity analysis. In Fig. 7, we examine the metadata prediction ability across the 5 blocks of ResNet50 models trained in diferent ways. All models reach a strong metadata prediction ability already after two blocks, since metadata traces are low-level signals easily captured by early layers [82]. The ability further improves after the third block for all models, but then diverges: it starts dropping in deeper layers for the model trained on the original IN1k, while continuing to improve for the model trained on the modified IN1k with strong correlations. This shows that the former learns to ignore metadata in deeper layers, delegating semantic feature extraction to them, whereas the latter learns to rely on metadata and cannot ignore it, suggesting that semantic discrimination is compromised in favor of metadata reliance. The use of mitigation efective augmentations, as identified in Sec. 4.1, reduces the MP accuracy. Interestingly, the model trained on strong JPEG-semantics correlations behaves similarly for resizing: it captures the resizing metadata in early layers and cannot ignore it in deeper ones, even though it was not trained with strong resizing-semantics correlations. This is further evidence that the model learns to rely on low-level traces in general, not only on the specific attribute strongly correlated with semantics during training.

![](images/bfff749cdac31d0b53e0d686dfcaf452da77b5a1fb539f784554ad6f6724c5d1.jpg)  
Fig. 8: t-SNE visualization of features before (top) and after (bottom) posthoc mitigation for two classes (•, ▲) from IN1k processed by diferent JPEG parameters (■, ■). Used models are ResNet50 trained on diferent versions of IN1k with manipulated JPEG labels from Sec. 3.2 (original IN1k/none, $p _ { i } = 8 0$ and $p _ { i } = 1 0 0$ with low, strong and maximal correlations, respectively).

Representation space visualization before/after post-hoc mitigation. Fig. 8 provides a t-SNE [38] visualization of ResNet50 features for two IN1k classes, where all images are processed twice with two diferent sets of JPEG parameters. The models are trained on the original IN1k and on modified versions with strong processing metadata-semantics correlations (Sec. 3.2), shown before and after post-hoc mitigation. For the model trained on the original IN1k, the classes are decently separated while points with diferent JPEG parameters overlap well; the small signs of misalignment are consistent with the quantitative measurements in Fig. 3, indicating mild sensitivity to metadata. The model thus learns semantic features and is only subtly afected by metadata. In contrast, the two classes start to mix for models trained under strong correlations, while points with diferent JPEG parameters become more separated, showing that the model relies on metadata instead of semantics and that metadata leaves strong traces in the representation space. At $p _ { i } = 1 0 0 \%$ , the classes are almost completely mixed, and the features can instead be separated by JPEG processing. Finally, post-hoc mitigation improves semantic class separability even for the model trained with $p _ { i } = 1 0 0 \%$ correlation.

Impact of mitigation on OOD tasks. Since metadata traces can be a spurious low-level signal that models exploit as a shortcut, reducing reliance on them should be most beneficial under distribution shift, where such shortcuts no longer hold. We therefore investigate whether reducing the models’ metadata sensitivity improves performance on common out-of-distribution (OOD) ImageNet test sets: ImageNet-C [20], ImageNet-R [19], and ImageNet-Sketch [70]. Fig. 9 reports results without and with the post-hoc mitigation. The mitigation helps in many cases, sometimes considerably, and the gains are largest for the models most sensitive to metadata (according to Fig. 6: CVLs and supervised ConvNeXt), while sometimes hurting less sensitive ones like DINOv2. We observe similar trends for ResNet50 trained on the original IN1k (ImageNet-C: 44.2 → 45.0, ImageNet-R: 34.5→37.7, ImageNet-Sketch: 26.4→30.0) when mitigating during training with augmentations.

![](images/d2277d53c1839ca22dc69d4290650619996136681768bed2c9148f8069a49d3a.jpg)  
Fig. 9: Post-hoc adversarial mitigation with a linear layer – impact on ImageNet OOD test sets. Performance is reported for the original frozen foundational models and for the models adjusted by an additional linear layer trained to mitigate the metadata sensitivity guided by manipulated JPEG labels on IN1k. Accuracy reported on ImageNet-C instead of the commonly reported error.

## 6 Related work

## 6.1 Shortcut learning

In empirical risk minimization (ERM) settings, neural networks have been shown to rely on simple input features as shortcuts to solve the task being learned, even if the shortcuts are only spuriously correlated with features more relevant to the task [14, 66]. This is because under ERM, the objective is simply to minimize error, meaning that shortcut features will be learned as long as they help with the training objective [42, 57]. Not only are shortcut features learned, but they are even preferred by models due to their simplicity. Even in the presence of features that are more salient and more informative, neural networks will still rely more strongly on the simple spurious features if the salient features are more complex to learn [62].

A frequent origin of shortcuts is dataset bias. Due to data sources and collection methods [35,63], as well as their properties such as confounding factors [81], data scarcity [48], collection location [13], and collection timeframe [53], resulting datasets may exhibit unique characteristics to the point that it is possible to distinguish them from one another despite having similar design goals [35, 68]. More importantly, among these idiosyncrasies may be correlations that create exploitable shortcuts. While these commonly manifest as object-background correlations, as is the case with ImageNet [21,45,64], e.g. the side of freight cars being associated with grafiti, non-object-centric correlations also exist. In LAION, images of cardboard boxes are commonly accompanied by watermarks [31]. Even spatial characteristics, such as object location, and lower-level features, such as color distributions, have been found to be correlated with diferent semantic concepts [23, 41]. Models trained on these datasets in standard ERM settings typically learn these shortcuts and predict the semantic label even when the image contains only the shortcut feature [21, 31, 45, 64].

In contrast to the aforementioned immediately perceptible cues, we focus on largely imperceptible shortcuts, mainly JPEG compression artifacts. Previous studies that have examined their role in shortcut learning focus on specialized tasks [16] or exist outside the context of their correlation with semantic labels [9], while our work investigates their presence in large general-purpose training datasets and consequent ability to interfere with foundation models.

Additionally, neural networks have been shown to rely on frequency-based shortcuts [73, 74], e.g. zebra patterns being easily distinguishable in the spectrum, and such shortcuts arise even in models trained at ImageNet scale. Although some types of metadata could also introduce frequency-related shortcuts, these works do not account for them.

## 6.2 Shortcut mitigation

A diverse range of strategies has been proposed to mitigate shortcut learning. One prominent direction is data-centric, where the training distribution is modified to reduce spurious correlations. These approaches include data augmentation techniques [15, 49, 79], such as image stylization or mixup, which expand the dataset with semantically consistent transformations. Other methods rely on resampling or reweighting strategies [27, 30] to balance class distributions, as well as concept discovery and pseudo-label extraction techniques [44, 60, 76] that incorporate additional supervisory signals. A second line of work mitigates shortcuts directly during training. Some approaches introduce additional classification heads [1, 24] or train parallel models [59, 60, 71] to explicitly capture shortcut-related features, thereby encouraging the main model to ignore them.

Post-training approaches have also been explored, typically in a two-stage framework. These methods often leverage biased models to identify potential shortcuts [69, 77, 80], employ ensembles of bias models [3, 25, 29, 43], or focus on misclassified examples using contrastive learning techniques [85]. In addition, optimization-based methods modify the training objective to improve robustness to spurious correlations [32,33,56]. Other approaches fine-tune biased models either entirely [67] or only at the final layer [11, 22, 26, 50], often using feature reweighting strategies. Recent work has also begun exploring bias mitigation in foundation models, particularly in vision–language systems [69,78]. Our mitigation strategy draws inspiration from these directions. Our post-hoc contrastive separation term is similar to prior work [1, 85], but unlike prior methods, our approach specifically targets metadata traces in vision encoders.

## 7 Discussion and conclusions

We study why vision encoders become sensitive to image metadata and show that this sensitivity is the result of shortcut learning, emerging when metadata correlates with the semantic supervision used during pretraining. This accounts for its presence under categorical supervision on ImageNet and, as our controlled LAION experiments indicate, under caption supervision in vision-language pretraining. Self-supervised methods such as the DINO family are less sensitive, since they do not optimize for semantic targets, yet not fully invariant, as metadata still serves as a shortcut for matching views of the same image. More broadly, we establish the image formation and processing pipeline itself as a source of bias, imperceptible to humans yet readily exploited by models.

Crucially, metadata sensitivity is not purely a disadvantage. The tendency to capture pixel-level signal that harms robustness is what makes an encoder a strong detector of generated images: models trained under stronger metadatasemantics correlations, and thus more sensitive, detect synthetic images noticeably better. This reframes recent observations that frozen CLIP features excel at fake-image detection as a direct consequence of metadata encoding, making sensitivity double-edged, a disadvantage for semantic generalization but an asset for forensics. In our analysis we identify further connections than detection: even images synthesized by Stable Difusion, which never pass through a physical acquisition pipeline, seem to carry pixel-level metadata traces correlated with semantics, suggesting generators implicitly imitate the acquisition and processing signatures of their training data.

Since metadata distributions drift across cameras, capture conditions, and processing pipelines, encoders relying on such shortcuts are fragile precisely where deployment demands robustness. We show this sensitivity can be mitigated both during and after pretraining, generalizing across attributes while preserving semantic utility. Consistently, reducing it improves out-of-distribution generalization on ImageNet-C, ImageNet-R, and ImageNet-Sketch, with larger gains for the more sensitive models, suggesting metadata invariance may need to be a first-class objective when training or adapting foundation encoders.

We believe this work opens a new horizon, shifting attention from visible, semantic biases to the invisible traces left by the imaging pipeline. It remains open how strongly metadata sensitivity contaminates downstream tasks and the benchmarks used to evaluate encoders.

Acknowledgments: This work was supported by the GACR grant GA26- 24228S, the Horizon MSCA-PF grant No. 101154126, the Czech Technical University in Prague grant No. SGS26/074/OHK3/1T/13, the JSPS KAKENHI No. 23H00497 and No. 22K12091. We acknowledge VSB – Technical University of Ostrava, IT4Innovations National Supercomputing Center, Czech Republic, for awarding this project (OPEN-35-7) access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium through the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (grant ID: 90254).

## References

1. Alvi, M., Zisserman, A., Nellåker, C.: Turning a blind eye: Explicit removal of biases and variation from deep neural network embeddings. In: ECCV (2018)

2. Arjovsky, M., Bottou, L., Gulrajani, I., Lopez-Paz, D.: Invariant risk minimization. arXiv (2019)

3. Bahng, H., Chun, S., Yun, S., Choo, J., Oh, S.J.: Learning de-biased representations with biased representations. In: ICML (2020)

4. Beery, S., Van Horn, G., Perona, P.: Recognition in terra incognita. In: ECCV (2018)

5. Beyer, L., Izmailov, P., Kolesnikov, A., Caron, M., Kornblith, S., Zhai, X., Minderer, M., Tschannen, M., Alabdulmohsin, I., Pavetic, F.: FlexiViT: One model for all patch sizes. In: CVPR (2023)

6. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: ICCV (2021)

7. Cramér, H.: Mathematical methods of statistics. Princeton university press (1946)

8. Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.V.: RandAugment: Practical automated data augmentation with a reduced search space. In: CVPRW (2020)

9. Doersch, C., Gupta, A., Efros, A.A.: Unsupervised visual representation learning by context prediction. In: ICCV (2015)

10. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2020)

11. Du, M., Mukherjee, S., Wang, G., Tang, R., Awadallah, A., Hu, X.: Fairness via representation neutralization. In: NeurIPS (2021)

12. Fan, L., Chen, K., Krishnan, D., Katabi, D., Isola, P., Tian, Y.: Scaling laws of synthetic images for model training ... for now. In: CVPR (2024)

13. Gaviria Rojas, W., Diamos, S., Kini, K., Kanter, D., Janapa Reddi, V., Coleman, C.: The dollar street dataset: Images representing the geographic and socioeconomic diversity of the world. NeurIPS (2022)

14. Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A.: Shortcut learning in deep neural networks. Nature Machine Intelligence (2020)

15. Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F.A., Brendel, W.: ImageNet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In: ICLR (2018)

16. Grommelt, P., Weiss, L., Pfreundt, F.J., Keuper, J.: Fake or jpeg? revealing common biases in generated image detection datasets. In: ECCVW (2024)

17. Grootendorst, M.: BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794 (2022)

18. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (2016)

19. Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., et al.: The many faces of robustness: A critical analysis of out-of-distribution generalization. In: CVPR (2021)

20. Hendrycks, D., Dietterich, T.: Benchmarking neural network robustness to common corruptions and perturbations. In: ICLR (2019)

21. Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., Song, D.: Natural adversarial examples. In: CVPR (2021)

22. Izmailov, P., Kirichenko, P., Gruver, N., Wilson, A.G.: On feature learning in the presence of spurious correlations. In: NeurIPS (2022)

23. Jahanian\*, A., Chai\*, L., Isola, P.: On the "steerability" of generative adversarial networks. In: ICLR (2020)

24. Kim, B., Kim, H., Kim, K., Kim, S., Kim, J.: Learning not to learn: Training deep neural networks with biased data. In: CVPR (2019)

25. Kim, N., Hwang, S., Ahn, S., Park, J., Kwak, S.: Learning debiased classifier with biased committee. In: NeurIPS (2022)

26. Kirichenko, P., Izmailov, P., Wilson, A.G.: Last layer re-training is suficient for robustness to spurious correlations. In: ICLR (2022)

27. LaBonte, T., Hill, J., Zhang, X., Muthukumar, V., Kumar, A.: The group robustness is in the details: Revisiting finetuning under spurious correlations. In: NeurIPS (2024)

28. LAION: Releasing re-laion-5b: transparent iteration on laion-5b with additional safety fixes. https://laion.ai/blog/relaion-5b/ (2024)

29. Lee, Y., Yao, H., Finn, C.: Diversify and disambiguate: Out-of-distribution robustness via disagreement. In: ICLR (2023)

30. Li, Y., Vasconcelos, N.: Repair: Removing representation bias by dataset resampling. In: CVPR (2019)

31. Li, Z., Evtimov, I., Gordo, A., Hazirbas, C., Hassner, T., Ferrer, C.C., Xu, C., Ibrahim, M.: A whac-a-mole dilemma: Shortcuts come in multiples where mitigating one amplifies others. In: CVPR (2023)

32. Liu, E.Z., Haghgoo, B., Chen, A.S., Raghunathan, A., Koh, P.W., Sagawa, S., Liang, P., Finn, C.: Just train twice: Improving group robustness without training group information. In: ICML (2021)

33. Liu, S., Zhang, X., Sekhar, N., Wu, Y., Singhal, P., Fernandez-Granda, C.: Avoiding spurious correlations via logit correction. In: ICLR (2022)

34. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: ICCV (2021)

35. Liu, Z., He, K.: A decade’s battle on dataset bias: Are we there yet? In: ICLR (2025)

36. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: CVPR (2022)

37. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: ICLR (2019)

38. Van der Maaten, L., Hinton, G.: Visualizing data using t-sne. JMLR (2008)

39. McInnes, L., Healy, J., Astels, S.: hdbscan: Hierarchical density based clustering. The Journal of Open Source Software 2(11), 205 (2017)

40. McInnes, L., Healy, J., Melville, J.: UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv preprint arXiv:1802.03426 (2018)

41. Meister, N., Zhao, D., Wang, A., Ramaswamy, V.V., Fong, R., Russakovsky, O.: Gender Artifacts in Visual Datasets. In: ICCV (2023)

42. Nagarajan, V., Andreassen, A., Neyshabur, B.: Understanding the failure modes of out-of-distribution generalization. In: ICLR (2021)

43. Nam, J., Cha, H., Ahn, S., Lee, J., Shin, J.: Learning from failure: Training debiased classifier from biased classifier. In: NeurIPS (2020)

44. Nam, J., Kim, J., Lee, J., Shin, J.: Spread spurious attribute: Improving worstgroup accuracy with spurious attribute estimation. In: ICLR (2022)

45. Neuhaus, Y., Augustin, M., Boreiko, V., Hein, M.: Spurious features everywhere - large-scale detection of harmful spurious features in ImageNet. In: ICCV (2023)

46. Ojha, U., Li, Y., Lee, Y.J.: Towards universal fake image detectors that generalize across generative models. In: CVPR (2023)

47. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: DINOv2: Learning robust visual features without supervision. TMLR (2024)

48. Parashar, S., Lin, Z., Liu, T., Dong, X., Li, Y., Ramanan, D., Caverlee, J., Kong, S.: The neglected tails in vision-language models. In: CVPR (2024)

49. Puli, A.M., Joshi, N., Wald, Y., He, H., Ranganath, R.: Nuisances via negativa: Adjusting for spurious correlations via data augmentation. TMLR (2024)

50. Qiu, S., Potapczynski, A., Izmailov, P., Wilson, A.G.: Simple and fast group robustness by automatic feature reweighting. In: ICML (2023)

51. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: ICML (2021)

52. Ramos, R., Stojnić, V., Kordopatis-Zilos, G., Nakashima, Y., Tolias, G., Garcia, N.: Processing and acquisition traces in visual encoders: What does clip know about your camera? In: ICCV (2025)

53. Recht, B., Roelofs, R., Schmidt, L., Shankar, V.: Do ImageNet classifiers generalize to ImageNet? In: ICML (2019)

54. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: CVPR (2022)

55. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M.S., Berg, A.C., Fei-Fei, L.: ImageNet large scale visual recognition challenge. IJCV (2015)

56. Sagawa, S., Koh, P.W., Hashimoto, T.B., Liang, P.: Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In: ICLR (2020)

57. Sagawa, S., Raghunathan, A., Koh, P.W., Liang, P.: An investigation of why overparameterization exacerbates spurious correlations. In: ICML (2020)

58. Sariyildiz, M.B., Alahari, K., Larlus, D., Kalantidis, Y.: Fake it till you make it: Learning transferable representations from synthetic ImageNet clones. In: CVPR (2023)

59. Sarridis, I., Koutlis, C., Papadopoulos, S., Diou, C.: Flac: Fairness-aware representation learning by suppressing attribute-class associations. PAMI (2024)

60. Sarridis, I., Koutlis, C., Papadopoulos, S., Diou, C.: MAVias: Mitigate any visual bias. In: ICCV (2025)

61. Sha, Z., Li, Z., Yu, N., Zhang, Y.: De-fake: Detection and attribution of fake images generated by text-to-image generation models. In: SIGSAC (2023)

62. Shah, H., Tamuly, K., Raghunathan, A., Jain, P., Netrapalli, P.: The pitfalls of simplicity bias in neural networks. NeurIPS (2020)

63. Shirali, A., Hardt, M.: What makes ImageNet look unlike LAION. TMLR (2025)

64. Singla, S., Feizi, S.: Salient ImageNet: How to discover spurious features in deep learning? In: ICLR (2022)

65. Smeu, S., Oneata, E., Oneata, D.: Declip: Decoding clip representations for deepfake localization. In: WACV (2025)

66. Suhail, P., Sethi, A.: Shortcut Learning Susceptibility in Vision Classifiers. In: Workshop on Spurious Correlation and Shortcut Learning: Foundations and Solutions (2025)

67. Taghanaki, S.A., Khani, A., Khani, F., Gholami, A., Tran, L., Mahdavi-Amiri, A., Hamarneh, G.: Masktune: mitigating spurious correlations by forcing to explore. In: NeurIPS (2022)

68. Torralba, A., Efros, A.A.: Unbiased look at dataset bias. In: CVPR (2011)

69. Varma, M., Delbrouck, J.B., Chen, Z., Chaudhari, A., Langlotz, C.: Ravl: Discovering and mitigating spurious correlations in fine-tuned vision-language models. In: NeurIPS (2024)

70. Wang, H., Ge, S., Lipton, Z.C., Xing, E.P.: Learning robust global representations by penalizing local predictive power. In: NeurIPS (2019)

71. Wang, H., He, Z., Lipton, Z.C., Xing, E.P.: Learning robust representations by projecting superficial statistics out. In: ICLR (2019)

72. Wang, Q., Lin, Y., Chen, Y., Schmidt, L., Han, B., Zhang, T.: A sober look at the robustness of clips to spurious features. NeurIPS (2024)

73. Wang, S., Veldhuis, R.N.J., Brune, C., Strisciuglio, N.: What do neural networks learn in image classification? A frequency shortcut perspective. In: ICCV (2023)

74. Wang, S., Veldhuis, R.N.J., Strisciuglio, N.: Do imagenet-trained models learn shortcuts? the impact of frequency shortcuts on generalization. In: CVPR (2025)

75. Wightman, R., Touvron, H., Jégou, H.: Resnet strikes back: An improved training procedure in timm. arXiv (2021)

76. Wu, S., Yuksekgonul, M., Zhang, L., Zou, J.: Discover and cure: Concept-aware mitigation of spurious correlation. In: ICML (2023)

77. Yang, Y., Gan, E., Dziugaite, G.K., Mirzasoleiman, B.: Identifying spurious biases early in training through the lens of simplicity bias. In: AISTATS (2024)

78. Yang, Y., Nushi, B., Palangi, H., Mirzasoleiman, B.: Mitigating spurious correlations in multi-modal models during fine-tuning. In: ICML (2023)

79. Yao, H., Wang, Y., Li, S., Zhang, L., Liang, W., Zou, J., Finn, C.: Improving out-of-distribution robustness via selective augmentation. In: ICML (2022)

80. Yenamandra, S., Ramesh, P., Prabhu, V., Hofman, J.: Facts: First amplify correlations and then slice to discover bias. In: ICCV (2023)

81. Zech, J.R., Badgeley, M.A., Liu, M., Costa, A.B., Titano, J.J., Oermann, E.K.: Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLoS Medicine 15(11), e1002683 (2018)

82. Zeiler, M.D., Fergus, R.: Visualizing and understanding convolutional networks. In: ECCV (2014)

83. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: ICCV (2023)

84. Zhang, H., Cissé, M., Dauphin, Y.N., Lopez-Paz, D.: mixup: Beyond empirical risk minimization. In: ICLR (2018)

85. Zhang, M., Sohoni, N.S., Zhang, H.R., Finn, C., Ré, C.: Correct-n-contrast: A contrastive approach for improving robustness to spurious correlations. In: NeurIPS (2022)

## Supplementary Material

![](images/79b105c1e29204127d51d8f5a28609b03a53c94e3e14211d05c81969fc86a675.jpg)

![](images/c5fb82a1ce85f0d497e82fd4789fa73aaa5154b4522614b3a0cfda157277a90e.jpg)

![](images/992a29a1036e3b0bcb3695873a5c5e548ee9c214ca90707d72d4c8c7c60e64ca.jpg)

![](images/bb1711f8226ff4039bee0c77e011f18dc321f5c741edeb009c2f6f770860f99c.jpg)

![](images/c109d0afe393962f84a9cf445b7704e98ad6b43df57bdd05dd1b10a44b5a6683.jpg)

![](images/50ca26faa4bba7ae4a6c354c169637d24e931a9acd9f4d6e5b464077f4c72a8a.jpg)

![](images/5a6cae1128c5bb35d6f0b027f3a9f87aa08a08402098091dcabf19d755c577a9.jpg)

![](images/73dab19c8f1d444f479ca39684e6528f4f9f5e047e001fefa6496e8dbb9b60de.jpg)  
Fig. I: Impact of correlations between semantics and processing metadata on the model’s sensitivity to metadata in a class controlled setting. ResNet50 trained on diferent versions of IN1k with manipulated JPEG labels. We use probability $p _ { c }$ to control the fraction of classes that exhibit correlations between JPEG labels and semantic labels, instead of using $p _ { i }$ shown in Fig. 3. The larger the $p _ { c } ,$ , the stronger the correlation between JPEG labels and semantic labels. None: model trained on original IN1k. Diference between $p _ { c } = 0$ and none: the former has no JPEG-semantics correlations, while the latter has the original IN1k correlations. SP: two types of test sets are used, the original one (none, green color) which is fixed across values of $p _ { c } .$ and 7 diferent versions (purple color) created by manipulating the JPEG labels of the original test set in the same way as the corresponding training set for each value of $p _ { c }$

## A Class controlled correlations of semantics and processing metadata

In Sec. 3.2, we use probability $p _ { i }$ to control the percentage of images of each class that exibit correlations of semantics and processing metadata. We now take a diferent approach and we randomly assign one specific metadata label only to a fraction $p _ { c }$ of classes. Images belonging to the remaining classes are randomly assigned to their metadata label. Thus, $p _ { c }$ controls the proportion of classes exhibiting correlations of semantics and processing metadata.

Fig. I presents the results of this experiment, where we observe similar trends as in Fig. 3 (Sec. 3.2), that the models trained on data with stronger correlations between metadata and semantic labels are more sensitive to metadata.

Additionally, among the two ways $( p _ { i }$ and $p _ { c } )$ of creating correlations, the one that controls the number of correlated classes seems to have a stronger efect on the model’s ability to encode metadata information and on the sensitivity of semantic prediction abilities to metadata. This suggests that having whole classes exhibiting correlation between metadata and semantics has a stronger efect on the model’s ability to encode metadata information than having only a fraction of images of each class exhibiting such correlation.

![](images/712b431aad5a6cc8cd13097bf1be689be0eef4aed7f12800d7f651daebaf25c4.jpg)

![](images/9b4b8b1d25190b51a792289c6fdddd7a56c8b1c89d5bdaa0b9475c1fff9698c9.jpg)

![](images/128bb92f89b47530283c78a63ba1326febf5f2a3af39dcb4f8041148f5416d57.jpg)

![](images/b6109d470021dfa05dc5aa67c352948c4eab130da8447a2eb5adc739f12723ba.jpg)

![](images/611e7b6005833646ea0322359af76f2a7e185a1c06db5feab1e7ab7122da9bd4.jpg)

![](images/b6a9ad76f4d9e08b83572975b82ddb2da8f2cee16d20896c3c13b555bbbc16d1.jpg)

![](images/2a3fef53761bfadb6e72cd18614d3721f31ece967854977ad594ca7dfb3ff22c.jpg)

![](images/2c29a354d76793fe058ecc0c11586702ed21d2d3fa2b1f58926bd287514f6a17.jpg)  
Fig. II: Impact of correlations between semantics and processing metadata on the model’s sensitivity to metadata. ResNet50 trained on diferent versions of IN1k with manipulated resize labels instead of JPEG labels in Fig. 3. The larger the $p _ { i }$ , the stronger the correlation between resize labels and semantic labels. None: model trained on original IN1k. Diference between $p _ { i } = 0$ and none: the former has no resize-semantics correlations, while the latter has the original IN1k correlations. SP: two types of test sets are used, the original one (none, green color) which is fixed across values of $p _ { i }$ , and 7 diferent versions (purple color) created by manipulating the resize labels of the original test set in the same way as the corresponding training set for each value of $p _ { i }$

## B Controlling correlations of semantics and processing metadata using resize

In Sec. 3.2, we introduce correlations of semantics and processing metadata by recompressing images based on their semantic label. To show that our observations are not dependent on the choice of processing metadata, instead of recompressing, we resize images based on their semantic label. We define 13 metadata labels that control the amount of upsampling or downsampling of the original image. All images are then resized according to one of these labels, creating a correlation between semantic labels and metadata labels. Fig. II presents the results of this experiment. We observe that models trained on data with stronger correlations between metadata and semantic labels are more sensitive to metadata.

![](images/74712644a08dc107d0d9e84bc30f995afd9c1606ca15414f51212a86c2e0f21f.jpg)

![](images/3947fc46c1d5616a898421dd92b697eea16830f311102f193438de74f550e544.jpg)

![](images/6aa55482f8aff52bcc4ceb12cfc83debe8743c8372a476703a3f27abc718a3be.jpg)

![](images/036e3e3a776f41419347fcb2c8994596727baddcd9733ea0c8adbdc667366ea3.jpg)

![](images/8c0bf13fd0e882f9613e9c12a1d9a37986714a21dae04a2ea9d344795533b38a.jpg)

![](images/ec7c522dc4f28069091e89a4aa9b6bfa3f768442fd931c94e44132ef7ca8f949.jpg)

![](images/88726f740172682e9cee99cd9a31bb0788af5916b5994c6ff9c4c8581ac1228f.jpg)  
<sup>p</sup><sub>i</sub>

![](images/a2c8b95ce58b0ee69602c2ca75abb1d3c15a7413582d1d679ce8045c02aa0431.jpg)  
<sup>p</sup><sub>i</sub>  
Fig. III: Impact of correlations between semantics and processing metadata on the model’s sensitivity to metadata. Diagnostic metrics for additional metadata attributes compared to Fig. 3 ResNet50 trained on diferent versions of IN1k with manipulated JPEG labels. The larger the $p _ { i }$ , the stronger the correlation between JPEG labels and semantic labels. None: model trained on original IN1k. Diference between $p _ { i } = 0$ and none: the former has no JPEG-semantics correlations, while the latter has the original IN1k correlations. SP: two types of test sets are used, the original one (none, green color) which is fixed across values of $p _ { i }$ , and 7 diferent versions (purple color) created by manipulating the JPEG labels of the original test set in the same way as the corresponding training set for each value of $p _ { i }$

## C Diagnostic metrics for additional metadata attributes

In the main paper, we evaluate models on two processing attributes (JPEG and resizing) and two acquisition attributes (make and aperture). Here, we present results for other processing and acquisition attributes from the benchmark by Ramos et al. [52]. Fig. III presents the results for the ResNet50 model trained on the data with correlations between JPEG labels and semantics labels (Sec. 3.2) for additional metadata attributes. We observe that models trained on data with stronger correlations are more sensitive to metadata, showing that our observations are not dependent on the metadata attribute used during evaluation. Additionally, in Fig. IV, we present the results of our post-hoc adversarial mitigation approach (Sec. 4.2) when applied to these other metadata attributes. We observe that our mitigation approach reduces sensitivity to metadata, demonstrating that it generalizes across metadata attributes.

![](images/f6223cb9e7a606c44ec12e3502d4f42eebd4144f56be697eccdc28741f8fb8cb.jpg)  
Fig. IV: Post-hoc adversarial mitigation with a linear layer – impact on model sensitivity to metadata. Diagnostic metrics for additional metadata attributes compared to Fig. 6.

## D Comparison of mitigation approaches

The two proposed mitigation approaches are applicable in diferent settings and assumptions: augmentations during the original model training and post-hoc on pre-trained models. However, the results in Tab. I indicate they complement each other, while performing roughly the same if used separately.

## E Ablation of the post-hoc adversarial mitigation

Fig. V presents the results of an ablation study of the post-hoc adversarial mitigation loss. We observe that both terms of the loss related to metadata mitigation $( \beta , \gamma )$ contribute to reducing sensitivity to metadata. Interestingly, we observe that the feature preservation term (α) does not significantly afect the semantic prediction (SP). However, we attribute this to the fact that we use a simple linear layer as the mitigation component. When using a more complex mitigation component such as MLP, we observe that this term becomes crucial for preserving semantic information.

<table><tr><td>aug post-hoc</td><td></td><td>MPp - JPEG  $\mathrm { M P _ { p } }$ </td><td>- Resizing</td><td> $\mathrm { M P a }$ </td><td>- Make MPa - Aperture</td><td>SP</td><td>SPDp - JPEG SPDp - Resizing</td><td></td><td> $\mathrm { { S P D a } }$  - Camera</td></tr><tr><td>x</td><td>x</td><td>37.3</td><td>63.4</td><td>23.8</td><td>11.9</td><td>78.2</td><td>2.7</td><td>4.2</td><td>3.5</td></tr><tr><td>√</td><td>x</td><td>30.7</td><td>53.1</td><td>21.4</td><td>11.1</td><td>77.3</td><td>1.4</td><td>1.8</td><td>1.6</td></tr><tr><td>x</td><td>√√</td><td>31.2</td><td>56.1</td><td>19.4</td><td>11.4</td><td>77.2</td><td>0.8</td><td>2.3</td><td>2.3</td></tr><tr><td>√</td><td></td><td>24.7</td><td>46.9</td><td>18.9</td><td>10.5</td><td>76.5</td><td>0.3</td><td>0.7</td><td>1.6</td></tr></table>

Table I: Comparison and combination of the mitigation methods. ResNet50 trained on original IN1k.

![](images/e0479b14bfd0f29f026755a94392e3a47f2b683f398120fda280f41cf96ed1d9.jpg)  
Fig. V: Impact of loss terms of the post-hoc adversarial mitigation. Mitigation applied on top of pretrained supervised ConvNeXt-B [36].

## F Implementation details

## F.1 Diagnostic metrics

For diagnostic metrics, we follow the setup of Ramos et al. [52] for all processing and acquisition attributes. An overview of the used processing and acquisition attributes, along with the number of classes in each case, is given in Tab. II; the full details can be found in the original paper [52].

<table><tr><td>Attribute</td><td>Type</td><td>Metric</td><td>Num. classes</td><td>Description</td></tr><tr><td>JPEG</td><td>processing</td><td>MP, SPD</td><td>6</td><td>amount of JPEG compression</td></tr><tr><td>Resizing</td><td>processing</td><td>MP, SPD</td><td>3</td><td>amount of image resizing</td></tr><tr><td>Sharpening</td><td>processing</td><td>MP, SPD</td><td>3</td><td>amount of sharpening</td></tr><tr><td>Interpolation</td><td>processing</td><td>MP, SPD</td><td>4</td><td>type of interpolation during resizing</td></tr><tr><td>Make</td><td>acquisition</td><td>MP</td><td>9</td><td>manufacturer of the camera</td></tr><tr><td>Aperture</td><td>acquisition</td><td>MP</td><td>17</td><td>size of the opening in the lens</td></tr><tr><td>Model</td><td>acquisition</td><td>MP</td><td>88</td><td>specific camera model used</td></tr><tr><td>Exposure</td><td>acquisition</td><td>MP</td><td>16</td><td>amount of light captured by sensor</td></tr><tr><td>ISO speed</td><td>acquisition</td><td>MP</td><td>16</td><td>camera sensor&#x27;s sensitivity to light</td></tr><tr><td>Focal length</td><td>acquisition</td><td>MP</td><td>13</td><td>light convergence strength of lens</td></tr><tr><td>Camera</td><td>acquisition</td><td>SPD</td><td>2</td><td>type of camera used</td></tr></table>

Table II: Image processing and acquisition attributes used in diagnostic metrics.

![](images/f96b78ff52652cd7dd15f15b8afe0a12e4fb1ea194088a31dbbde1419782509a.jpg)

![](images/c7bdfd7ad0c4ea0601ce568f5812901ab0b312e8ab54e18f7e99999df8e07121.jpg)  
Fig. VI: Topic proportions within the Re-LAION-2B subsets with baseline, stronger, weaker correlations, discussed in Sec 3.4. Left: the values plotted at the same x-value correspond to the same topic. Topics are sorted in descending order by the proportion in the baseline subset. Right: the values plotted at the same x-value correspond to the same topic rank. Proportions are sorted in descending order.

## F.2 Training on IN1k

We train all of our models for 300 epochs on IN1k. For ResNet50, we follow the A2 recipe of Wightman et al. [75]. ViT-S/16 and ViT-B/16 are trained using the AdamW [37] optimizer with a batch size of 4096 and a cosine-scheduled learning rate of 1e − 3, with a warmup of 40 and 32 epochs, respectively. We additionally utilize 0.1 weight decay, 0.1 dropout, RandomAugment [8], and 0.5 mixup [84].

## F.3 Training on Re-LAION-2B

We use BERTopic [17] for topic extraction, where the captions are embedded with a sentence encoder<sup>6</sup>, then dimensionality-reduced via UMAP [40] and clustered into topics via HDBSCAN [39]. Topic embeddings are then calculated by averaging the embeddings of the assigned captions, which is used to categorize new captions via cosine similarity. By first fitting our topic model on 1M captions, then applying the provided auto topic reduction method, we obtain 6, 238 topics. The proportions of topic representation are shown in Fig. VI. In general, topics are represented consistently across the three subsets, with topics highly represented in the baseline subset also being among the most represented topics in the other subsets.

All of our models are trained for 32 epochs using the AdamW [37] optimizer with a batch size of 320 and a cosine-scheduled learning rate of $5 e - 4 ,$ , with a warmup of 10, 000 warmup steps. We additionally utilize 0.2 weight decay and automatic mixed precision.

## F.4 Details about synthesizing correlation between semantics and processing metadata

We introduce controlled correlation between semantics and processing metadata in IN1k by either JPEG compression or resizing.

JPEG compression: We consider a set ${ \mathcal { I } } = { \mathcal { Q } } \times { \mathcal { C } } = \{ ( q _ { 1 } , c _ { 1 } ) , \cdot \cdot \cdot , ( q _ { J } , c _ { J } ) \}$ of J diferent JPEG metadata labels defined by the quality $q \in \mathcal { Q }$ and chroma subsampling $c \in { \mathcal { C } } .$ . We use $\mathcal { Q } = \{ 7 5 , 7 7 , \cdot \cdot \cdot , 9 5 \}$ and $\mathcal { C } = \{ 4 { : } 2 { : } 0 , 4 { : } 4 { : } 4 \}$ , which yields $J = 2 2$ distinct values of JPEG metadata labels.

Resizing: We consider a set $\mathcal { R } = \{ r _ { 1 } , \cdot \cdot \cdot , r _ { R } \} = \{ \textstyle { \frac { 1 } { 4 } } , \cdot \cdot \cdot , \frac { 1 } { 1 . 5 } , 1 , 1 . 5 , \cdot \cdot \cdot , 4 \}$ where $r _ { i }$ defines the change in image area while preserving the aspect ratio.

## F.5 Post-hoc adversarial mitigation

Training of our post-hoc mitigation model proceeds in two steps, with each step optimized for 10 epochs using the AdamW [37] optimizer with 1.0 learning rate.

1. Train the metadata prediction model h while keeping the mitigation component frozen by optimizing cross-entropy loss with supervision from metadata labels.

2. Train the mitigation component f while keeping the metadata prediction model frozen by optimizing a loss

$$
\begin{array} { r l } {  { \mathcal { L } = - \underbrace { \frac { \alpha } { N } \sum _ { i = 1 } ^ { N } \log ( \sigma ( \frac { f ( x _ { i } ) ^ { \top } x _ { i } } { \tau } ) ) } _ { \mathrm { k e p r e d s e t ~ c o t i n a l ~ f e a t u r e } } } \quad } & { } \\ & { \underbrace { - \frac { \beta } { N ^ { 2 } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \prod [ m ( x _ { i } ) = m ( x _ { j } ) \wedge i \neq j ] \log ( 1 - \sigma ( \frac { f ( x _ { i } ) ^ { \top } f ( x _ { j } ) } { \tau } ) ) } _ { \mathrm { p u s h ~ a p a r t ~ t e n t u r e s ~ t h a t e ~ c h a r c e ~ t h e ~ s a m e ~ m e c t a d a t a i ~ l a b e l } } } \\ & { + \underbrace { \frac { \gamma } { N } \sum _ { i = 1 } ^ { N } \mathrm { E x } [ \mathrm { 1 } \cdot ( 1 / M \parallel h ( f ( x _ { i } ) ) ) , } _ { \mathrm { p r e d u c t ~ u n i t e r m ~ d i s t r i b u t i o n } } , } \end{array}\tag{1}
$$

where $\alpha , \beta ,$ and $\gamma$ are weights, σ is a sigmoid function, $\tau$ is a temperature, I is an indicator function, m(x) outputs the metadata label associated with $x ,$ KL is KL-divergence, and 1 is a vector of all ones.

We initialize f to an identity, while we initialize h randomly. We fix $\alpha = 4 0$ $\beta = 2 0 , \gamma = 1 . 0 , \tau = 0 . 1$ , and repeat the steps $L = 1 0$ times.

![](images/7f029565b84d44ab40ad6cc07c51e0ef4e7adc00ff49ad4ccab95320410f7da7.jpg)  
Fig. VII: Impact of correlations between semantics and processing metadata on the model’s sensitivity to metadata. ViT-S/16, instead of ResNet50 $\left( \mathrm { F i g . 3 } \right)$ trained on diferent versions of IN1k with manipulated JPEG labels. The larger the $p _ { i } .$ the stronger the correlation between JPEG labels and semantic labels. None: model trained on original IN1k. Diference between $p _ { i } ~ = ~ 0$ and none: the former has no JPEG-semantics correlations, while the latter has the original IN1k correlations. SP: two types of test sets are used, the original one (none, green color) which is fixed across values of $p _ { i } .$ , and 7 diferent versions (purple color) created by manipulating the JPEG labels of the original test set in the same way as the corresponding training set for each value of $p _ { i }$

## G Experiments with other backbones

In Figs. VII and VIII, we present the results of controlling correlations between semantics and processing metadata, as in Sec. 3.2, but for ViT-S/16 and ViT-B/16, respectively, instead of ResNet50. We observe that the introduction of JPEG-semantics correlation during training makes the model more sensitive to metadata, same as we observed for ResNet50 in Fig. 3. This shows that our observations are not model-dependent and that models of diferent architectures exhibit the same behaviour.

Additionally, Figs. IX and X show results parallel to our experiment in Sec. 3.4, where we train a model using CLIP loss [51] on subsets of Re-LAION-2B with diferent amount of correlations between semantics and acquisition metadata. This time we use ViT-S/16 and ConvNeXt-T backbones, instead of ResNet50. Results are consistent, as training with stronger correlations between semantics and acquisition metadata leads to worse SPD and SP, with the opposite being true for training with weaker correlations.

![](images/4e4bd8ab84f28643b684c5b5c7307c03f2e530746f66dc7d5319918474afc32e.jpg)

![](images/8836bf263eb4428bd43740dac40776a807f37a51bdb652219dfeb6174a6434f2.jpg)

![](images/f6b98e061626f1367b5f411a0e9182ceb98ee11793de340606e568f435c37c9c.jpg)

![](images/f8e82350116049f709a46a968a6887f26c6882ce8a349135d104ca0e52bb5ed2.jpg)

![](images/5337a909859f6034734fc2ad3bbe216e5b0c2fd5e299be47cd82ecaae0a0e83a.jpg)

![](images/ccda4b898e1c81a33433705f544f576fdf00b0525f2c31e48aac58fedc5b9c4c.jpg)  
<sup>p</sup><sub>i</sub>

![](images/d743b01d831bf047b7589c8e9fe800fc7458cb0f1d6c20cbab94c11178852418.jpg)  
<sup>p</sup><sub>i</sub>

![](images/22efa83a67cc1920c5c9d7f8b898de119b7a8b509d3d345fed04dd602f0f6785.jpg)  
<sup>p</sup><sub>i</sub>  
Fig. VIII: Impact of correlations between semantics and processing metadata on the model’s sensitivity to metadata. ViT-B/16, instead of ResNet50 (Fig. 3), trained on diferent versions of IN1k with manipulated JPEG labels. The larger the $p _ { i } ,$ the stronger the correlation between JPEG labels and semantic labels. None: model trained on original IN1k. Diference between $p _ { i } = 0$ and none: the former has no JPEG-semantics correlations, while the latter has the original IN1k correlations. SP: two types of test sets are used, the original one (none, green color) which is fixed across values of $p _ { i } ,$ and 7 diferent versions (purple color) created by manipulating the JPEG labels of the original test set in the same way as the corresponding training set for each value of $p _ { i }$

![](images/970e9d6c11ea997c1dcbcb32f8df230c3e931d799b843a6b4110e17453a8803f.jpg)

![](images/1d4b10f65c436be1762f083e94bd091846295b8b783c3e65f20e154354927661.jpg)

![](images/c488ffc57327e3b7b1dc426fc9fc9f0958861d2ab398fb39b7f56556d8ccaa35.jpg)

![](images/9129c698bde5f16daa4499c4f1f8ff438660bc64d7ac371e93a0b1ac259cad91.jpg)

![](images/5598a6a24c0a9e429695ce7a09ac36d98e382cd8ad945daece60ad94f92f9840.jpg)

![](images/c8a037b4db3dfa3a6bf87ee0a4217e37515c5a933456235c74d819e3c5c9f3ec.jpg)

![](images/158e7f9c5525a1ada2994914e423eec4c5d5292437c82baf9df032118fe5b4de.jpg)

![](images/aa0846c113d4c6805e65877c899bbad6b1c0fcea7325a633cc75d5bae3b3f272.jpg)  
Fig. IX: Impact of correlations between semantics and acquisition metadata on the model’s sensitivity to metadata. $\mathrm { V i T - S } / 1 6$ , instead of ResNet50 (Fig. 4), trained from scratch using CLIP loss [51] on three Re-LAION-2B subsets.

![](images/2489b36738ebf78705368574049fdd8d9d0968c10874fee87b1883861d41e988.jpg)

![](images/9a3a18826e4d085b7ba0d8cfdb9d7c21222e613a0cc06bbc21c5b031b92a0e65.jpg)

![](images/ee23aa3b310eab870110124007e6f1d5ea92397ce1e438a11e72c48dcb9bc08b.jpg)

![](images/f6a61ea2cbff018279d9d1fa64096cb928644571c20c380b6ee2442314f3b0b7.jpg)

![](images/dbf8db6a1e7dd12f5f975578e3a7d6c694af6159a2d93892a77c53e7d33eda0e.jpg)

![](images/afcfdc0373de9a7308a5dfbb10bcf5bc7dba213519e2dd02a7026bf3c6886dc1.jpg)

![](images/a83382a19f94e155bfaefab027ebba96aa58a00652d2575206e01818134e6327.jpg)

![](images/ff062d4d5fdb6c157a1dfaa435fc7d862a006074da87b46a812daabbff8c23b5.jpg)  
Fig. X: Impact of correlations between semantics and acquisition metadata on the model’s sensitivity to metadata. ConvNeXt-T, instead of ResNet50 (Fig. 4), trained from scratch using CLIP loss [51] on three Re-LAION-2B subsets.