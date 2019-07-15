<!-- 
footer: @imflash217 | flashAI.labs | 2017-2019 
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

<!-- fit --> Machine Learning

<!---
![bg](https://raw.githubusercontent.com/vinaykumar2491/marpX/master/assets/flashSlides.jpg)
-->

---
![bg right:50%](https://images.unsplash.com/photo-1514302240736-b1fee5985889?ixlib=rb-1.2.1&auto=format&fit=crop&w=1934&q=80)

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
        DT_FULL._remove(node)           # in-place removal
    else:
        continue
```

---

---

==Render inline== math such as $ax^2+bc+c$.

$$ I_{xx}=\int\int_Ry^2f(x,y)\cdot{}dydx $$

$$
f(x) = \int_{-\infty}^\infty
    \hat f(\xi)\,e^{2 \pi i \xi x}
    \,d\xi
$$

---

```python
class WowMagic:
    def __init__(self, magic):
        self.magic = magic
        self.authenticity = True
    def foolPeople(response):
        if response:
            continue
        else:
            pass
```

---
```css
%reset-button {
  border: none;
  margin: 0;
  padding: 0;
  width: auto;
  overflow: visible;

  background: transparent;

  /* inherit font & color from ancestor */
  font: inherit;
  color: inherit;

  /* Normalize `line-height`. Cannot be changed from `normal` in Firefox 4+. */
  line-height: normal;

  /* Corrects font smoothing for webkit */
  -webkit-font-smoothing: inherit;
  -moz-osx-font-smoothing: inherit;

  /* Corrects inability to style clickable `input` types in iOS */
  -webkit-appearance: none;

  // /* Remove excess padding and border in Firefox 4+ */
  &::-moz-focus-inner {
    border: 0;
    padding: 0;
  }
}

```

---

```javascript

module.exports = {
  ogImage: process.env.URL && `${process.env.URL}/og-image.jpg`,
  url: process.env.URL,
}

module.exports = {
  ogImage: process.env.URL && `${process.env.URL}/og-image.jpg`,
  url: process.env.URL,
}

module.exports = {
  ogImage: process.env.URL && `${process.env.URL}/og-image.jpg`,
  url: process.env.URL,
}

module.exports = {
  ogImage: process.env.URL && `${process.env.URL}/og-image.jpg`,
  url: process.env.URL,
}

module.exports = {
  ogImage: process.env.URL && `${process.env.URL}/og-image.jpg`,
  url: process.env.URL,
}

module.exports = {
  ogImage: process.env.URL && `${process.env.URL}/og-image.jpg`,
  url: process.env.URL,
}

module.exports = {
  ogImage: process.env.URL && `${process.env.URL}/og-image.jpg`,
  url: process.env.URL,
}
```

---

# ![](images/marp.png)





##### Markdown presentation writer, powered by [Electron](http://electron.atom.io/)



###### Created by Yuki Hattori ( [@yhatt](https://github.com/yhatt)  )



---

# Features



**Slides are written in Markdown.**
- Cross-platform. Supports Windows, Mac, and Linux
- Live Preview with 3 modes
- Slide themes (`default`, `gaia`) and custom background images
- Supports emoji :heart:
- Render maths in your slides
- Export your slides to PDF


---



# How to write slides?


Split slides by horizontal ruler `---`. It's very simple.

```md
# Slide 1
foobar
```

---



# Slide 2

foobar

**Notice: Ruler (`<hr>`) is not displayed in Marp.**


---


# Directives


Marp's Markdown has extended directives to affect slides.

Insert HTML comment as below:

```html
<!-- {directive_name}: {value} -->
```

```html
<!--
{first_directive_name}:  {value}
{second_directive_name}: {value}
...
-->
```
---
<!-- fit -->
## Global Directives


### `$theme`

Changes the theme of all the slides in the deck. You can also change from `View -> Theme` menu.

```
<!-- $theme: gaia -->
```

|Theme name|Value|Directive|
|:-:|:-:|:-|
|***Default***|default|`<!-- $theme: default -->`
|**Gaia**|gaia|`<!-- $theme: gaia -->`

---

### `$width` / `$height`

Changes width and height of all the slides.

You can use units: `px` (default), `cm`, `mm`, `in`, `pt`, and `pc`.

```html
<!-- $width: 12in -->
```

### `$size`

Changes slide size by presets.

Presets: `4:3`, `16:9`, `A0`-`A8`, `B0`-`B8` and suffix of `-portrait`.

```html
<!-- $size: 16:9 -->
```

<!--
$size: a4

Example is here. Global Directive is enabled in anywhere.
It apply the latest value if you write multiple same Global Directives.
-->

---

## Page Directives

The page directive would apply to the  **current page and the following pages**.
You should insert it *at the top* to apply it to all slides.

### `page_number`

Set `true` to show page number on slides. *See lower right!*

```html
<!-- page_number: true -->
```

<!--
page_number: true

Example is here. Pagination starts from this page.
If you use multi-line comment, directives should write to each new lines.
-->

---

### `template`

Set to use template of theme.

The `template` directive just enables that using theme supports templates.

```html
<!--
$theme: gaia
template: invert
-->
```

Example: Set "invert" template of Gaia theme.

---

### `footer`

Add a footer to the current slide and all of the following slides

```html
<!-- footer: This is a footer -->
```

Example: Adds "This is a footer" in the bottom of each slide

---

### `prerender`

Pre-renders a slide, which can prevent issues with very large background images.

```html
<!-- prerender: true -->
```

---

## Pro Tips

#### Apply page directive to current slide only

Page directive can be selectively applied to the current slide by prefixing the page directive with `*`.

```
<!-- *page_number: false -->
<!-- *template: invert -->
```

<!--
*page_number: false

Example is here.
Page number is not shown in current page, but it's shown on later pages.
-->

---

#### Slide background Images

You can set an image as a slide background.

```html
![bg](mybackground.png)
```

Options can be provided after `bg`, for example `![bg original](path)`.

Options include:

- `original` to include the image without any effects
- `x%` to include the  image at `x` percent of the slide size
- Include multiple`![bg](path)` tags to stack background images horizontally.
![bg](images/background.png)
---

#### Maths Typesetting
Mathematics is typeset using the `KaTeX` package. Use `$` for inline maths, such as $ax^2+bc+c$, and `$$` for block maths:
$$I_{xx}=\int\int_Ry^2f(x,y)\cdot{}dydx$$

```html

This is inline: $ax^2+bx+c$, and this is block:
$$I_{xx}=\int\int_Ry^2f(x,y)\cdot{}dydx$$


---

## Enjoy writing slides! :+1:

### https://github.com/yhatt/marp

Copyright &copy; 2016 [Yuki Hattori](https://github.com/yhatt)
This software released under the [MIT License](https://github.com/yhatt/marp/blob/master/LICENSE).
a
asa
aasdasd
asas
asds
ddd
a
adadd

addd

a
d
ddda
dad
adad
