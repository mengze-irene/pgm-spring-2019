---
layout: distill
title: Lecture 14 Markov Chain Monte Carlo 
description: An example of a distill-style lecture notes that showcases the main elements.
date: 2019-01-09

lecturers:
  - name: Eric Xing
    url: "https://www.cs.cmu.edu/~epxing/"

authors:
  - name: Irene Li
    url: "#"  # optional URL to the author's homepage
  - name: George Cai
    url: "#"
  - name: Zhuoran Zhang
    url: "#"
  - name: Jiacheng Zhu

editors:
  - name: Editor 1  # editor's full name
    url: "#"  # optional URL to the editor's homepage

abstract: >
  An example abstract block.
---

## Recap of Monte Carlo
### Monte Carlo methods are algorithms that:
* Generate samples from a given probability distribution $p(x)$.
* Estimate expectations of functions $E[f(x)]$ under a distribution $p(x)$.

### Why is Monte Carlo useful?
* Can use samples of $p(x)$ to approximate p(x) itself 
  * Allow us to do graphical model inference when we can't compute $p(x)$.
* Expectations $E[f(x)]$ reveal interesting properties about $p(x)$, e.g., means and variances of $p(x)$.

### Limitations of Monte Carlo
* Direct sampling
  * Hard to get rare events in high-dimensional spaces
  * Infeasible for MRFs, unless we know the normalizer $Z$.
* Rejection sampling, Importance sampling
  * Do not work well if the proposal $Q(x)$ is very different from $P(x)$.
  * Yet constructing a $Q(x)$ similar to $P(x)$ can be difficult.
    * Requires knowledge of the analytical form of $P(x)$ - but if we had that, we wouldn't even need to sample!
* Intuition: Instead of a fixed proposal $Q(x)$, use an adaptive proposal.

## Markov Chain Monte Carlo (MCMC)
MCMC algorithms feature adaptive proposals.
* Instead of $Q(x')$, use $Q(x'|x)$ where x' is the new state being sampled, and x is the previous sample.
* As x changes, $Q(x'|x)$ can also change (as a function of $x'$).

<figure>
<img src="{{ '/assets/img/notes/lecture-14/MCMC.png' | relative_url }}" />
<figcaption>
Comparison between using a fixed (bad) proposal and an adaptive proposal.
</figcaption>
</figure>

To understand how MCMC works, we need to look at Markov Chains first.

### Markov Chains
* A Markov Chain is a sequence of random variables $x^{(1)}$, $x^{(2)}$, ..., $x^{(n)}$ with the Markov Property

$$
P(x^{(n)}=x|x^{(1)}, ..., x^{(n-1)})=P(x^{(n)}=x|x^{(n-1)})
$$
  * $P(x^{(n)}=x|x^{(n-1)})$ is known as the transition kernel **transition kernel**.
  * The next state depends only on the preceding states.
  * Random variables $x^{(i)}$ can be **vectors**.
    * We define $x^{(i)}$ to be the t-th sample of **all** variables in a graphical model
    * $x^{(i)}$ represents the entire state of the graphical model at time $t$.

* We study homogeneous Markov Chains, in which the transition kernel $P(x^{(n)}=x|x^{(n-1)})$ is fixed with time.
  * To emphasize this, we will call the kernel $T(x'|x)$, where $x$ is the previous state and $x'$ is the next state.
   
### Markov Chains Concepts
Define a few important concepts of Markov Chains(MC)
* **Probability distribution over states**: $\pi^{(t)}(x)$ is a distribution over the state of the system $x$, at time $t$.
  * When dealing with MCs, we don't think of the system as being in one state, but as having a distribution over states.
  * For graphical models, remember that $x$ represents **all** variables.
* **Transitions**: recall that states transition from $x^{(t)}$ to $x^{(t+1)}$ according to the transition kernel $T(x'|x)$.      * We can also transition entire distributions: $\pi^{(t+1)}(x')=\sum_{x} \pi^{(t)}(x)T(x'|x)$
  * At time t, state x has probability mass $\pi^{(t)}(x)$. The transition probability redistributes this mass to other states $x’$.
* **Stationary distributions**: $\pi^{(t)}(x)$ is stationary if it does not change under the transition kernel:
$\pi(x')=\sum_{x} \pi(x)T(x'|x)$, for all $x'$. To understand stationary distributions, we need to define some notions:
  * **Irreducible**: an MC is irreducible if you can get from any state x to any other state x’ with probability > 0 in a finite number of steps, i.e., there are no unreachabble parts of the state space.
  * **Aperiodic**: an MC is aperiodic if you can return to any state x at any time.
    * Periodic MCs have states that need ≥2 time steps to return to (cycles).
  * **Ergodic (or regular)**: an MC is ergodic if it is irreducible and aperiodic
    * Ergodicity is important: it implies you can reach the stationary distribution $\pi_{st}(x)$, no matter the initial distribution $\pi^{(0)}(x)$.
    * All good MCMC algorithms must satisfy ergodicity, so that you can’t initialize in a way that will never converge.
  * **Reversible (detailed balance)**: an MC is reversible if there exists a distribution $\pi(x)$ such that the detailed balance condition is satisfied:
  $$\pi(x')T(x|x')=\pi(x)T(x'|x)$$
    Probability of $x'->x$ is the same as $x->x'$.
      * Reversible MCs **always** have a stationary distribution! Proof:
      <d-math block>
      \begin{aligned}
      \pi(x')T(x\ |x') & = \pi(x)T(x'\ | x) \\
      \sum_{x}\pi(x')T(x\ |x') & = \sum_{x}\pi(x)T(x'\ | x) \\
      \pi(x')\sum_{x}T(x\ |x') & = \sum_{x}\pi(x)T(x'\ | x) \\
      \pi(x') & = \sum_{x}\pi(x)T(x'\ | x) 
      \end{aligned}
      </d-math>
      Note that the last line is the definition of a stationary distribution!

## Metropolis-Hastings (MH) -- An MCMC method
### How the MH algorithm works in practice
1. Draws a sample x' from $Q(x'\ |x)$, where x is the previous sample.
2. The new sample x’ is accepted or rejected with some probability $A(x'\ | x)$
  * This acceptance probability is $A(x'\ | x)=min(1, \frac{P(x')Q(x\ |x')}{P(x)Q(x'\ |x)})$
  * $A(x'\ | x)$ is like a ratio of importance sampling weights
    * $P(x')/Q(x'\ |x)$ is the importance weight for x', $P(x)/Q(x\ |x')$ is the mportance weight for x.
    * We devide the importance wieght for x' by that of x
    * Notice that we only need to compute $P(x')/P(x)$ rather than $P(x')$ or $P(x)$ separately, so we don't need to know the normalizer.
  * $A(x'\ | x)$ ensures that, after sufficiently many draws, our samples will come from the true distribution $P(x)$.
 
<figure>
<img src="{{ '/assets/img/notes/lecture-14/MH_algo.png' | relative_url }}" />
<figcaption>
The Metropolis-Hastings Algorithm
</figcaption>
</figure>
 

### Why does Metropolis-Hastings work?
Since we draw a sample x' according to $Q(x'\ |x)$, and then accept/reject according to $A(x'\ |x)$, the transition kernel is:
$$T(x'\ |x)=Q(x'\ | x)A(x'\ |x)$$
We can prove that MH satisfies detailed balance/reversibility:
Recall that 
$$A(x'\ | x)=min(1, \frac{P(x')Q(x\ |x')}{P(x)Q(x'\ |x)})$$.

This implies the following:
$$\text{if} A(x'\ |x)<1 \text{then| \frac{P(x')Q(x\ |x')}{P(x)Q(x'\ |x)}>1 \text{and thus} A(x\ |x')=1$$

Now suppose $A(x'\ |x)<1$ and $A(x \ | x')=1$. We have:
<d-math block>
\begin{aligned}
A(x'\ |x) & = \frac{P(x')Q(x\ |x')}{P(x)Q(x'\ |x)} \\
P(x)Q(x'\ |x)A(x'\ |x) & = P(x')Q(x\ |x') \\
P(x)Q(x'\ |x)A(x'\ |x) & =P(x')Q(x\ |x')A(x\ |x') \\
\pi(x')T(x\ |x')& =\pi(x)T(x'\ |x)
\end{aligned}
</d-math>
The last line is exactly the detailed balance condition. 
In other words, the MH algorithm leads to a stationary distribution $P(x)$. Recall we defined $P(x)$ to be the true distribution of x. Thus, the MH algorithm eventually converges to the true distribution!

### Caveats
Although MH eventually converges to the true distribution $P(x)$, we have no guarantees as to when this will occur.
* MH has a "burn-in" period: an initial number of samples are thrown away because they are not from the true distribution.   
  * The burn-in period represents the un-converged part of the Markov Chain.
  * Knowing when to halt burn-in is an art. We will look at some techniques later in this lecture.

 
## Equations

This theme supports rendering beautiful math in inline and display modes using [KaTeX](https://khan.github.io/KaTeX/) engine.
You just need to surround your math expression with `$`, like `$ E = mc^2 $`.
If you leave it inside a paragraph, it will produce an inline expression, just like $E = mc^2$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

Alternatively and for more complex math environments, use `<d-math block>...</d-math>` tags.
Here is an example:

<d-math block>
\begin{aligned}
\left( \sum_{i=1}^n u_i v_i \right)^2 & \leq \left( \sum_{i=1}^n u_i^2 \right) \left( \sum_{i=1}^n v_i^2 \right) \\
\left| \int f(x) \overline{g(x)} dx \right|^2 & \leq  \int |f(x)|^2 dx \int |g(x)|^2 dx
\end{aligned}
</d-math>

Note that [KaTeX](https://khan.github.io/KaTeX/) is work in progress, so it does not support the full range of math expressions as, say, [MathJax](https://www.mathjax.org/).
Yet, it is [blazing fast](http://www.intmath.com/cg5/katex-mathjax-comparison.php).

***

## Figures

To add figures, use `<figure>...</figure>` tags.
Within the tags, define multiple rows of images using `<div class="row">...</div>`.
To add captions, use `<figcaption>...</figcaption>` tags.

Here is an example usage of a figure that consists of a row of images with a caption:

<figure id="example-figure" class="l-body-outset">
  <div class="row">
    <div class="col one">
      <img src="{{ 'assets/img/notes/template/1.jpg' | relative_url }}" />
    </div>
    <div class="col one">
      <img src="{{ 'assets/img/notes/template/2.jpg' | relative_url }}" />
    </div>
    <div class="col one">
      <img src="{{ 'assets/img/notes/template/3.jpg' | relative_url }}" />
    </div>
  </div>
  <figcaption>
    <strong>Figure caption title in bold.</strong>
    An example figure caption.
  </figcaption>
</figure>

Note that the figure uses `class="l-body-outset"` which lets it take more horizontal space.
For more on this, see layouts section below.
Also, the size of the images themselves is controlled by `class="one"`, `class="two"`, or `class="three"` which corresponds to 1/3, 2/3, 3/3 of the full horizontal space, respectively.

Here is the same example, but each image is captioned separately:
<figure id="example-figure" class="l-body-outset">
  <div class="row">
    <div class="col one">
      <img src="{{ 'assets/img/notes/template/1.jpg' | relative_url }}" />
      <figcaption>
        <strong>Figure caption title 1.</strong>
        Caption text for figure 1.
      </figcaption>
    </div>
    <div class="col one">
      <img src="{{ 'assets/img/notes/template/2.jpg' | relative_url }}" />
      <figcaption>
        <strong>Figure caption title 2.</strong>
        A very very very long caption text for figure 2 so that it is longer than the image itself.
      </figcaption>
    </div>
  </div>
</figure>

Here is an example that shows how the figures of different sizes are aligned:

<figure>
  <div class="row">
    <div class="col two">
      <img src="{{ 'assets/img/notes/template/4.jpg' | relative_url }}" />
    </div>
    <div class="col one">
      <img src="{{ 'assets/img/notes/template/2.jpg' | relative_url }}" />
      <figcaption>
        <strong>Subcaption.</strong>
        The content of the subcaption.
      </figcaption>
    </div>
  </div>
  <figcaption>
    <strong>The second row figure caption title.</strong>
    An example of a sencond row figure caption.
  </figcaption>
</figure>

***

## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

***

## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>

***

## Code Blocks

Syntax highlighting is provided within `<d-code>` tags.
An example of inline code snippets: `<d-code language="html">let x = 10;</d-code>`.
For larger blocks of code, add a `block` attribute:

<d-code block language="javascript">
  var x = 25;
  function(x) {
    return x * x;
  }
</d-code>

***

## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body` sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

***

## Arbitrary $$\LaTeX$$ (experimental)

In fact, you can write entire blocks of LaTeX using `<latex-js>...</latex-js>` tags.
Below is an example:<d-footnote>If you don't see anything, it means that your browser does not support Shadow DOM.</d-footnote>

<latex-js style="border: 1px dashed #aaa;">
This document will show most of the supported features of \LaTeX.js.

\section{Characters}

It is possible to input any UTF-8 character either directly or by character code
using one of the following:

\begin{itemize}
    \item \texttt{\textbackslash symbol\{"00A9\}}: \symbol{"00A9}
    \item \verb|\char"A9|: \char"A9
    \item \verb|^^A9 or ^^^^00A9|: ^^A9 or ^^^^00A9
\end{itemize}

\bigskip

\noindent
Special characters, like those:
\begin{center}
\$ \& \% \# \_ \{ \} \~{} \^{} \textbackslash % \< \>  \"   % TODO cannot be typeset
\end{center}
%
have to be escaped.

More than 200 symbols are accessible through macros. For instance: 30\,\textcelsius{} is
86\,\textdegree{}F.
</latex-js>

Note that you can easily interleave latex blocks with the standard markdown.

<latex-js style="border: 1px dashed #aaa;">
\section{Environments}

\subsection{Lists: Itemize, Enumerate, and Description}

The \texttt{itemize} environment is suitable for simple lists, the \texttt{enumerate} environment for
enumerated lists, and the \texttt{description} environment for descriptions.

\begin{enumerate}
    \item You can nest the list environments to your taste:
        \begin{itemize}
            \item But it might start to look silly.
            \item[-] With a dash.
        \end{itemize}
    \item Therefore remember: \label{remember}
        \begin{description}
            \item[Stupid] things will not become smart because they are in a list.
            \item[Smart] things, though, can be presented beautifully in a list.
        \end{description}
    \item Technical note: Viewing this in Chrome, however, will show too much vertical space
        at the end of a nested environment (see above). On top of that, margin collapsing for inline-block
        boxes is not allowed. Maybe using \texttt{dl} elements is too complicated for this and a simple nested
        \texttt{div} should be used instead.
\end{enumerate}
%
Lists can be deeply nested:
%
\begin{itemize}
  \item list text, level one
    \begin{itemize}
      \item list text, level two
        \begin{itemize}
          \item list text, level three

            And a new paragraph can be started, too.
            \begin{itemize}
              \item list text, level four

                And a new paragraph can be started, too.
                This is the maximum level.

              \item list text, level four
            \end{itemize}

          \item list text, level three
        \end{itemize}
      \item list text, level two
    \end{itemize}
  \item list text, level one
  \item list text, level one
\end{itemize}

\section{Mathematical Formulae}

Math is typeset using KaTeX. Inline math:
$
f(x) = \int_{-\infty}^\infty \hat f(\xi)\,e^{2 \pi i \xi x} \, d\xi
$
as well as display math is supported:
$$
f(n) = \begin{cases} \frac{n}{2}, & \text{if } n\text{ is even} \\ 3n+1, & \text{if } n\text{ is odd} \end{cases}
$$

</latex-js>

Full $$\LaTeX$$ blocks are supported through [LaTeX JS](https://latex.js.org/){:target="\_blank"} library, which is still under development and supports only limited functionality (which is still pretty cool!) and does not allow fine-grained control of the layout, fonts, etc.

*Note: We do not recommend using using LaTeX JS for writing lecture notes at this stage.*

***

## Print

Finally, you can easily get a PDF or printed version of the notes by simply hitting `ctrl+P` (or `⌘+P` on macOS).

