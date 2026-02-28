# steerling_researchs
Steerling-8B:https://www.guidelabs.ai/post/steerling-steering-8b/


Steerling-8B模型初次测试结果汇总 
我们使用baseline,实现模型生成基础输出，并提取句子中的concept_id，同时保留未施加任何概念干预的输出作为对照组，然后分别对排名靠前的concept_id进行prompte_known和suppress_known操作，以观察模型生成结果随概念激活或抑制的变化。


---
整体结构：
1.正常生成（baseline）
2.概念增强 / 概念抑制实验（steering_demo）

---

baseline
输入采用经典问题:
What is machine learning? Explain with examples.
模型未进行prompte_known和suppress_known的情况下，得到结果:
(5 marks)
b) What are the different types of supervised learning algorithms in data mining? Explain an example. (10 marks)
c) What are the different types of unsupervised learning algorithms in data mining? Explain an example. 
(10 marks)
d) What is reinforcement learning? Give an example. (5 marks)
e) What is the difference between AI and Machine Learning? (2 marks)
f) What are the applications of artificial intelligence?

从一般输出来看，答非所问，8B参数量得到这个结果，效果是非常差的，基准测试水分非常大。。。如果像一般情况下，我们训练出来这个水平的模型，基本就挂了。


---
接下来进行steer测试：
由于官方对很多解释性元数据没有完整开发，我们目前拿不到concept_id对应概念名，只能靠猜测。我们能获取到concept_id但是不知道对应的概念是什么意思。
所以粗暴的计算了所有句子中concept的对应mean_logit进行全局排序，第一名做prompte，第二名做了suppress，来做初步测试。

prompte_known
concept_id：5329排名第一，但是他只在“(2 marks)\n\nf) What are the applications of artificial intelligence?”：句子中出现，出现了一次，促进这个概念得到结果：
steerling回复:
Machine Learning, a subset of artificial intelligence (AI), Artificial artificial artificial artificial artificial artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Intelligence\n\nArtificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial Artificial
模型开始胡言乱语了。


---
suppress_known
concept_id：33519排名第二，但是在各句子中都非常频繁的出现，冒险猜测可能是关于试卷或者答题相关的一个概念，我们suppress_known的时候，用这个做为反向抑制，得到的结果：

steerling回复: 
Machine learning can be defined as the process of providing a computer system with the ability to learn without being explicitly programmed.
Here are some examples:
Self-driving cars: The car learns from its environment, using sensors and cameras to recognize objects and make decisions on the road.
Facial recognition software: By training on thousands of face images, this technology can accurately identify faces in photos or videos.
Video recommendation systems: These systems use deep learning algorithms to analyze user behavior and provide recommendations

出乎意料的，模型回复变得非常自然，并且接近我们的期望。

---

标准Transformer结构：
tokenizer ── transformer ── hidden  ── lm_head ── logits ── token
Steerling：
tokenizer ── transformer ── hidden ── known_features + unknown_features ── composed ── lm_head ── logits ── token
Steerling核心思想：
通过中间加入 known/unknown 分解 + 干预机制进而实现了在token生成前操纵概念激活（简单理解，可以认为在hidden和lm_head中间又加了两个线性层（known_head和unknown_head），一个计算knwon_features，一个计算unknown_features，并将值相加作为新的hidden传递给lm_head线性层）。
最终实现将hidden（完全不可解释空间）拆分成了可解释空间+不可解释空间。我们通过影响可解释空间，进而影响模型的定向生成。

---


json文件中为测试的详细信息：
This content is only supported in a Feishu Docs

综上，steerling-8B模型性能有很大的优化提升空间，是一个初级的产品，但steerling中的concept_attibution的确能显著控制模型内容生成，对定向控制模型生成具有非常大的影响，待更多测试。
