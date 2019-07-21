<!-- 
footer: @imflash217 | flashAI.labs | 2019 
-->

<style>
section {
    font-family: "Fira Code";
    font-size: 20px;
    background-image: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);
}

.language-javascript,
.language-python{
    font-family: "Fira Code";
}
</style>

<!---
![bg](https://raw.githubusercontent.com/vinaykumar2491/marpX/master/assets/flashSlides.jpg)
-->

---
# <!-- fit --> Machine Learning `Theory & Practice`
---
<!-- _header: Roadmap -->

- ML Foundation:
    - Function Approximation & Generalization Guarantees
    - Decision Tree Learning
    - Overfitting & Validation
    - Pruning Techniques `later`
    - Naive Bayes & MLE/MAP estimates `optional`
    - SVMs & Random Forests `optional`
- Neural Networks:
    - Linear Classification & Regression
    - Backpropagation
    - Feed-forward Deep Neural Networks (DNN)
    - Convolutional Neural Networks (CNN)
    - Overfitting & Cross-validation `revisted`
    - Various Weight Initialization Strategies
    - Recurrent NN, LSTMs & GRUs
    - CNN implemenation in `PyTorch`
    - LSTM implementation in `PyTorch`

---
<!-- _header: What is Machine Learning? -->
Three main parameters in understanding, designing & debugging any machine learning algorithm:
- Performane `P`
- Task `T`
- Experience `E`

So, any machine learning algorithm is merely just a **well defined learning task** $<P,T,E>$

Some examples we'll see: Emergency C-sections, Playing forecasts, etc.

---
<!-- _header: Function Approximation -->
Problem Setting:
- Set of possible instances $\bold{X}$
- Unknown target function $f: \bold{X}\rightarrow \bold{Y}$
- Set of function hypotheses $H = \{h|h:\bold{X}\rightarrow \bold{Y}\}$

Inputs:
- Training examples $\{<x_{i}, y_{i}>\}$ of unknown target function $f$

Output:
- Hypothesis $h \in H$ which best approximates $f$

---
<!-- header: Decision Trees -->
Problem Setting:
- Set of possible instances $\bold{X}$
    - Each Instance $x$ in $\bold{X}$ is a FEATURE VECTOR
    - $x = <x_{1}, x_{2}, x_{3} ... x_{n}>$
- Unknown target function $f: \bold{X}\rightarrow \bold{Y}$
    - $\bold{Y}$ is descrete valued
- Set of function hypotheses $H = \{h|h:\bold{X}\rightarrow \bold{Y}\}$
    - Each hypothesis $h$ is a DT

Inputs:
- Training examples $\{<x^{i}, y^{i}>\}$ of unknown target function $f$

Output:
- Hypothesis $h \in H$ which best approximates $f$

---
$f: \langle\text{OUTLOOK, TEMP, HUMIDITY, WIND}\rangle \rightarrow \text{PLAY}?$

```python
DAY     OUTLOOK     TEMP    HUMIDITY    WIND  | PLAY?
------------------------------------------------------
1       sunny       hot     high        weak    False
2       sunny       hot     high        strong  False
3       overcast    hot     high        weak    True
4       rain        mild    high        weak    True
5       rain        cool    normal      weak    True
6       rain        cool    normal      strong  False
7       overcast    cool    normal      strong  True
8       sunny       mild    high        weak    False
```

---
**Problem Setting:**
- Set of possible instances $\bold{X}$
    - Each Instance $x$ in $\bold{X}$ is a **FEATURE VECTOR**
    - For eg: $x = <Outlook=Sunny, Temp=Hot, Humidity=High, Wind=Weak>$
- Unknown target function $f: \bold{X}\rightarrow \bold{Y}$
    - $\bold{Y}$ is descrete valued
    - For eg: Here, $y^{i}=Yes$ if we play else $No$
- Set of function hypotheses $H = \{h|h:\bold{X}\rightarrow \bold{Y}\}$
    - Each hypothesis $h$ is a DT
    - Trees sort $x$ to leaf-node, which assigns $y$

**Inputs:**
- Training examples $\{<x^{i}, y^{i}>\}$ of unknown target function $f$

**Output:**
- Hypothesis $h \in H$ which best approximates $f$


---

```python
OUTLOOK = sunny:
|   HUMIDITY = high:    False
|   HUMIDITY = normal:  True
OUTLOOK = overcast:     True
OUTLOOK = rain:
|   WIND = strong:      False
|   WIND = weak:        True
```

**Strategy:**
- Internal Node: test one discrete-valued attribute $x_{i}$
- Brach from a node: select one value for $x_{i}$
- leaf node: predict $Y$ or ($P(Y|X \in leaf)$)

---
<!-- header: Decision Trees: Examples -->
- How would you represent following functions as DT?
    - $Y = X_{1}X_{2}$
    - $Y = X_{1} \vee X_{2}$
    - $Y = X_{1}X_{2} \vee X_{3}X_{4}(\neg X_{1})$

---
# Emergency C-sections risk prediction using DT
- 1000 medical records
- Negative examples are C-sections

```python
                                        [833+,  167-] 0.83+ 0.17-

Fetal_Presentation = 1:                 [822+,  116-] 0.88+ 0.12-
|   Previous_C-section = 0:             [767+,   81-] 0.90+ 0.10-
|   |   Primiparous = 0:                [399+,   13-] 0.97+ 0.03-
|   |   Primiparous = 1:                [368+,   68-] 0.84+ 0.16-
|   |   |   Fetal_Distress = 0:         [334+,   47-] 0.88+ 0.12-
|   |   |   |   Birth_weight <  3349g:  [201+, 10.6-] 0.95+ 0.05-
|   |   |   |   Birth_weight >= 3349g:  [113+, 36.4-] 0.78+ 0.22-
|   |   |   Fetal_Distress = 1:         [ 34+,   21-] 0.62+ 0.38-
|   Previous_C-section = 1:             [ 55+,   35-] 0.61+ 0.39-
Fetal_presentation = 2:                 [  3+,   29-] 0.11+ 0.89-
Fetal_Presentation = 3:                 [  8+,   22-] 0.27+ 0.73-
```

---
<!-- header: Top-Down Induction of DT (an ID3 approach)-->
```python
tree.root = ROOT
def loop(tree, node):
    node.next = A                   #The BEST decision attribute for next node based on some CRITERION
    for ex in examples:
        result = sort(ex, tree)     # returns a sequence of 0 & 1 (0=not_fully_sorted till leaf node)
    if 0 in result:
        loop(tree, node.next)
    else:
        break;

loop(tree, tree.root)
```

---
<!-- _header: Function Approximation: An ID3 search over heuristic-space -->
```
length(x_i) = n
y = {0, 1}
```
# How many data samples we need to see to create a fully accurate DT?

---
But there is a problem!!!
How do we select the **BEST attribute** for an node?? :thinking:
Which **CRITERION** we used to make this decision?

What are some possible **CRITERIA**??

How well can we define our problem (the ones we saw before)?
What was our **objective**? accuracy :thinking: error :thinking: or something else??

---
If we ponder carefully, we will notice that our _objective_ was to **classify the data as PURELY as possible**

This brings us to one of the most commonly known & widely used metric for classification **purity** called **ENTROPY**

NOTE: 
- Though there can be arguments in favour of _accuracy_ or _error_ metrics too, these provide less generalization-gurantees ('ll see later)
- Fundamentally, any CONCAVE function is well suited* to be used as a classification criterion ('ll see later) 

---
<!-- header: ENTROPY: "Defines how impure a group is." -->

$$
\displaystyle \mathcal{H}(X) = -\sum_{i=1}^{n}{P(X=i)\cdot\log_{2}{P(X=i)}}
$$

$\mathcal{H}(X)$ is the expected number of bits needed to encode a randomly drawn value of $X$ (under most efficient code).

Specific conditional Entropy:
$$
\displaystyle \mathcal{H}(X|Y=y) = -\sum_{i=1}^{n}{P(X=i|Y=y)\cdot\log_{2}{P(X=i|Y=y)}}
$$

Conditional Entropy:
$$
\displaystyle \mathcal{H}(X|Y) = -\sum_{y\in\text{values}(Y)}{P(Y=y)\cdot\mathcal{H}(X|Y=y)}
$$

Mutual Information (aka **Information Gain**):
$$
\displaystyle \mathcal{I}(X,Y) = \mathcal{H}(X)-\mathcal{H}(X|Y) = \mathcal{H}(Y)-\mathcal{H}(Y|X)
$$

$\mathcal{I}_{S}(F,Y)$ is the expected reduction in entropy of target variable $Y$ for data sample $S$ due to sorting on feature $F$
$$
\text{Gain}(S,F) = \mathcal{I}_{S}(F,Y) = \mathcal{H}_{S}(Y)-\mathcal{H}_{S}(Y|F)
$$

---
```python
                          S=[9+, 5-]    H(S)   = 0.940
HUMIDITY = high:        S_0=[3+, 4-]    H(S_0) = 0.985  fr_0 = (3+4)/(9+5) = 7/14
HUMIDITY = normal:      S_1=[6+, 1-]    H(S_1) = 0.592  fr_1 = (6+1)/(9+5) = 7/14
                        Gain(S, HUMIDITY) = H(S) - fr_0*H(S_0) - fr_1*H(S_1) = 0.151


                          S=[9+, 5-]    H(S)   = 0.940
WIND = weak:            S_0=[6+, 2-]    H(S_0) = 0.811  fr_0 = (6+2)/(9+5) = 8/14
WIND = strong:          S_1=[3+, 3-]    H(S_1) = 1.0    fr_1 = (3+3)/(9+5) = 6/14
                        Gain(S, WIND) = H(S) - fr_0*H(S_0) - fr_1*H(S_1) = 0.048

```
Higher the `Gain()`, the better. So `HUMIDITY` is selected as the node-feature for this particular node in the DT.
Similarly we run tests for all features and find that `OUTLOOK` is the best features at the `ROOT` of this DT.

---
<!-- header: Occam's Razor: "Simplest is the best, that's the nature's property" -->
ID3 perform heuristic search through space of Decision Trees. It stops at the _smallest acceptable tree_. **Why?**
##### Reason: Occam's Razor

But why to prefer _shorter hypothesis_?
- In favour: 
    - Fewer short hypothesis than longer ones
        - Shorter hypothesis that fits the data is less likely to be a **statistical coincidence**.
        - Complex hypothesis are more prone to **overfitting**.
- Against:
    - Fewer number of hypothesis with specific #nodes and specific attribute. SO, less probability to find a solution
    - Why should we prefer _short hypothesis_? What's special about those?
    - No mathemetical gurantees.

---
<!-- header: Overfitting -->
### How do we justify that the DT that is built using some training data will generalize over unseen data aswell?
### What is the metric to measure it?

### <!-- fit --> Overfitting
### But how do we measure "overfitting"?

---
Consider a _hypothesis_ $h$ and its:
- Error over **train** data: $\text{error}_{\text{train}}(h)$
- **True** error over **all** data: $\text{error}_{\text{true}}(h)$

We say that $h$ **overfits the train data** if:
$$
\text{error}_{\text{true}}(h) \gt \text{error}_{\text{train}}(h)
$$
and the **amount of overfitting** is:
$$
\delta_{\text{overfit}} = \text{error}_{\text{true}}(h) - \text{error}_{\text{train}}(h)
$$

But there is another way to understand it.
# Demo with an DT example

---

```python
DAY     OUTLOOK     TEMP    HUMIDITY    WIND  | PLAY?
------------------------------------------------------
1       sunny       hot     high        weak    False
2       sunny       hot     high        strong  False
3       overcast    hot     high        weak    True
4       rain        mild    high        weak    True
5       rain        cool    normal      weak    True
6       rain        cool    normal      strong  False
7       overcast    cool    normal      strong  True
8       sunny       mild    high        weak    False
------------------------------------------------------
9       sunny       cool    high        weak    True
```

```python
OUTLOOK = sunny:
|   HUMIDITY = high:
|   |   TEMP = hot:     False
|   |   TEMP = mild:    False
|   |   TEMP = cool:    True
|   HUMIDITY = normal:  True
OUTLOOK = overcast:     True
OUTLOOK = rain:
|   WIND = strong:      False
|   WIND = weak:        True
```

---
## Cross-Validation

```python
-----------------------------   --------------
| Overall Training data     |   |  Test data |
-----------------------------   --------------

-----------------------------   --------------
|  fold1  |  fold2  | fold3 |   |  Test data |
-----------------------------   --------------

-----------------------------   --------------
|  train  |  train  |  val  |   |  Test data |
-----------------------------   --------------
-----------------------------   --------------
|  train  |   val   | train |   |  Test data |
-----------------------------   --------------
-----------------------------   --------------
|   val   |  train  | train |   |  Test data |
-----------------------------   --------------
```
Because when we use some data to build a hypothesis/DT/NN, we can gurantee that it will perform better (when tested) against `train` as compared to `val` or `test` data. 
So, $\text{error}_{\text{val}} \gt \text{error}_{\text{train}}$ always.
Hence in this methodology, we say that overfiting happens when $\text{error}_{\text{val}}$ **increases continuously**.

---
<!-- header: Overfitting: How to avoid it? -->
# How can we avoid overfitting train data?
- Stop growing the DT when data split not **statistically important**
- Grow full tree the **post-prune**

### Reduced Error pruning:
```python
TRAIN_SET, VAL_SET, TEST_SET            # defining train/val/test dataset
DT_FULL                                 # the DT that classifies TRAIN_SET correctly

for node in DT_FULL.nodes:
    if error(DT_FULL.remove(node), VAL_SET) <= error(DT_FULL, VAL_SET):     # notice the equal sign here
        DT_FULL._remove(node)           # in-place GREEDY removal
    else:
        continue
```
- Provides smallest version of the most accurate subtree

---
# <!-- fit --> What do we do if data is limited? :thinking:
# <!-- fit --> `Data-augmentation!!`

---
![bg width:80%](https://bair.berkeley.edu/static/blog/data_aug/basic_aug.png)

---
<!-- header: Data-driven approaches -->

- k-Nearest Neighbors (kNN)
- Linear Classification & Optimization
- Backpropagation
- Neaural Networks

## Task: Visual Recognition

---
<!-- header: The Problem Statement -->
- Given: 
    - set of descrete `labels`
    - a RGB image `(3D-matrix of numbers from 0-256)`
- Task: 
    - Classify the given-image into one of the given-discrete-labels

### Seems straight-forward...Doesn't it? Then what's the big problem?

---
# <!-- fit --> Semantic Gap

- Viewpoint variation : `The pixel values (3D-matrix) changes as the camera moves`
- Background clutter
- Illumination
- Deformation
- Occlusion
- Intraclass variation

# By now, we all should be able to appreciate the `gravity` :worried: of the problem we're trying to solve.

---
![bg vertical width:80%](https://raw.githubusercontent.com/vinaykumar2491/marpX/master/assets/cat1.png)
![bg width:80%](https://raw.githubusercontent.com/vinaykumar2491/marpX/master/assets/cat2.png)
![bg width:80%](https://raw.githubusercontent.com/vinaykumar2491/marpX/master/assets/cat3.png)
![bg width:80%](https://raw.githubusercontent.com/vinaykumar2491/marpX/master/assets/cat4.png)
![bg height:80%](https://raw.githubusercontent.com/vinaykumar2491/marpX/master/assets/cat5.png)

---
## And similar problem persists with `text` or `audio` data (infact various forms of data) `can you name a few more?`

```
What's up?
What's up dude?
What's going on?
Hey buddy!! what is on!!?
Yo bro!! wassup!!
Kya haal-chaal hai bhai?
Sab thik-thak?
...
...
```

All these above texts convey more-or-less the same intent. 
But the wordings are so much different.
Our algorithm (or neuralNet) must learn to understand that intent 
(we call it `sentiment` or `emotion` detection).

### Same follows with `audio` data too..
Can you name a few scenarios in audio-context?

---
<!-- header: Attempts made to tackle this problem -->
```python
def classify(input_data):
    # some magic code
    # some more magic code
    # maybe some more :P
    return class_label
```
That's what a traditional engineer will do in 2000s!!
But this won't be enough to properly tackle the challenges discussed. How come not?

# <!-- fit --> Any suggestions to tackle this?

---
- Attempts made:
    - `image/video` Edge detection
    - `image/video` Corner detection
    - `audio` Speaker-specific speech detection wsing GMM, HMM etc
    - `text` Traditional NLP using GMM, HMM, hot-encodings etc.

Engineers have tried to solve these complex problems since 1950s using neuralNets :wow:
- Hubel-weisel's cat experiment
- Rosenblatt's perceptron

But failed terribly due to lack of one very very important part of solution.

---
Courtesy to Tim-Berners-Lee's `www.` and enormous increase in number of sensors; now we have `big-data` too.
## <!-- fit --> data

---
<!-- header: Data-driven approach -->
One of the very primary big-data came in the form of `CIFAR10` dataset
- 10 classes `(car, airplane, horse, dog, cat ....)`
- 50000 training images (RGB)
- 10000 test images (RGB)

https://www.cs.toronto.edu/~kriz/cifar.html

We'll use this dataset for most of our examples. Lets see our 1st `data-driven` approach.

Just like `Entropy` in our Decison Trees, we need a metric to establish relationship within our data?

---
<!-- header: Nearest Neighors: L1-distance (manhattan distance) -->
# L1 distance (manhattan distance)
`pixel-wise absolute value differences`
$$
\mathcal{d}_{1}(\mathcal{I}_{1}, \mathcal{I}_{2}) = \sum_{p}\left|\mathcal{I}^{p}_{1}-\mathcal{I}^{p}_{2}\right|
$$
```python
import numpy as np

def metric_L1(image1: np.ndarray, image2: np.ndarray) -> np.int:
    assert image1.shape == image2.shape
    L1_distance = 0
    for i in np.arange(image1.shape[0]):
        for j in np.arange(image1.shape[1]):
            for k in np.arange(image1.shape[2]):
                L1_distance += np.abs(image1[i,j,k] - image2[i,j,k])
    return L1_distance

##----------------------------------------------------------------------------##

def metric_L1_vectorized(image1: np.ndarray, image2: np.ndarray) -> np.int:
    assert image1.shape == image2.shape
    return np.sum(np.abs(np.add(image1, (-1)*image2)))
```

---
<!-- header: Nearest Neighbors -->

```python
import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X: np.ndarray, Y: np.ndarray):
        """
        X   : np.shape(N, D)   : Each row is an example data (image)
        Y   : np.shape(N,)     : The labels for each example data
        """
        # remembering the training data
        self.X_train = X
        self.Y_train = Y

    def predict(self, X_test) -> np.ndarray:
        """
        X_test  : np.shape(N, D): Each row is an example we wish to predict for
        returns : np.shape(N,)  : Predicted label for each of the N test samples
        """
        n_test = X_test.shape[0]
        Y_pred = np.zeros(n_test, dtype=self.Y_train.dtype)                     # matching the output types of train & test labels

        # finding the nearest training image using the L1 distance
        for i in np.arange(n_test):
            distances = np.sum(np.abs(self.X_train - X_test[i, :]), axis=1)     # distances.shape = self.Y_train.shape
            nn_idx = np.argmin(distances)                                       # nearest neighbor index
            Y_pred[i] = self.Y_train[nn_idx])                                   # getting the appropriate label

        return Y_pred
```
---
Is this a good algorithm? Let's see:
    - How fast is the `training` process?
    - How fast is the `test/prediction` process?

---
Is this a good algorithm? Let's see for `N` train examples:
    - How fast is the `training` process?           `O(1)`
    - How fast is the `test/prediction` process?    `O(N)`

Is it desirable property of a good algorithm?

---
We should design algorithm that behaves like:
# <!-- fit --> `t_test << t_train`



---

---

---
<!-- header: References -->





















