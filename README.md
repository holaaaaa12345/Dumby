
<h4>p value and False Positive Rate FPR</h4>

The one sample t-test will produce a $p$ value. By definition, $p$ value is the probability 
of obtaining at least as extreme as a statistic given $h_{0}$ is true. If the $p$ value of a statistic 
is below 5%, i.e. when it is in the __critical region__, then the $h_{0}$ can be rejected. Consequently, __the FPR__, 
the probability that the t test falsely rejects the $h_{0}$ when the it is actually true, 
should equal the probability of obtaining statistic inside the critical region. Therefore, under the 
correct use of t-test, given all assumptions satisfied, __the FPR should always be 5%__. But what if 
those assumptions are not all satisfied?

<h4>Normality Assumption and FPR</h4>

This simulation will satisfy all but the normality assumption. If normality is violated, the FPR can 
still be 5% provided that the sample size is large. However, how large that sample size should be varies 
depending on the distribution. This simulation lets you choose between 6 different distributions to draw 
sample from, specify their parameters, pick with the sample size, and see the resulting FPR.

<h4>The Monte Carlo Simulation and FPR</h4>

This program will take $1,000,000$ independent samples from a chosen distribution with a chosen sample 
size and then perform one sample t test on them with the $h_{0}$ being true, meaning the value to test 
the difference from is the true mean of the distribution. It will then divide the number of significant 
$p$ values by $1,000,000$. This last quantity is, by definition, the FPR.

<h4>Purpose</h4>

This program has no practical benefit whatsoever :stuck_out_tongue_closed_eyes:. 
However, you can get a good sense of the normality assumption and an intuition for the robustness, 
the degree to which its assumption can be violated without sacrificing accuracy, of one sample t test.

