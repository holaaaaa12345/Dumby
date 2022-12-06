import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("seaborn-dark")
import scipy.stats
from PIL import Image



class Beta():

    def input_parameter(self):
        st.markdown(r"$\Large\alpha$")
        self.alpha = st.number_input(label="None", value=0.5, step=0.5, 
                                     label_visibility="collapsed")
        st.markdown(r"$\Large\beta$")
        self.beta = st.number_input(label="None", value=0.7, step=0.5, 
                                    label_visibility="collapsed")
        self.scipy_object = scipy.stats.beta(a=self.alpha, b=self.beta)
        self.mean = self.scipy_object.mean()
    
    def validate_parameter(self):
        if self.alpha <= 0 or self.beta <=0:
            raise ValueError

    def get_axes(self):
        x_axis = np.linspace(0, 1, 50)
        y_axis = self.scipy_object.pdf(x_axis)
        return x_axis, y_axis
        
    def show_function(self):
        st.markdown(r"""$f(x; \alpha,\beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1}
                    (1 - x)^{\beta - 1}$ where $B(\alpha, \beta) = \int_0^1 t^{\alpha - 1}
                    (1 - t)^{\beta - 1} dt$""")

    def get_sample(self, n):
        return np.random.beta(a=self.alpha, b=self.beta, size=n)


class Exponential():

    def input_parameter(self):
        st.markdown(r"$\Large\beta$")
        self.beta = st.number_input(label="None", value=1.0, step=0.5, 
                                    label_visibility="collapsed")
        self.scipy_object = scipy.stats.expon(loc=0, scale=self.beta)
        self.mean = self.scipy_object.mean()
        self.s_dev = self.scipy_object.std()
    
    def validate_parameter(self):
        if self.beta <= 0:
            raise ValueError

    def get_axes(self):
        reach = 4 * self.s_dev
        x_axis = np.linspace(0, self.mean + reach, 50)
        y_axis = self.scipy_object.pdf(x_axis)
        return x_axis, y_axis
        
    def show_function(self):
        st.latex(r"""f(x; \frac{1}{\beta}) = \frac{1}{\beta} 
                 e^{-\frac{x}{\beta}}""")

    def get_sample(self, n):
        return np.random.exponential(self.beta, size=n)


class Uniform():

    def input_parameter(self):
        st.markdown(r"$\Large{a:}$")
        self.a = st.number_input(label="None", value=0.0, step=0.5, 
                                 label_visibility="collapsed")
        st.markdown(r"$\Large{b:}$")
        self.b = st.number_input(label="None", value=2.0, step=0.5, 
                                 label_visibility="collapsed")
        self.scipy_object = scipy.stats.uniform(self.a, self.b-self.a)
        self.mean = self.scipy_object.mean()
        self.s_dev = self.scipy_object.std()
    
    def validate_parameter(self):
        if self.a > self.b:
            raise ValueError
        
    def get_axes(self):
        x_axis = np.array([self.a, self.b])
        y_axis = self.scipy_object.pdf(x_axis)
        return x_axis, y_axis
        
    def show_function(self):
        st.latex(r"""f(x; a,b) = {\begin{cases}{\frac {1}{b-a}}&{\text{for }}x\in [a,b]\\
                 0&{\text{otherwise}}\end{cases}}""")

    def get_sample(self, n):
        return np.random.uniform(self.a, self.b, size=n)


class SkewNormal():
    def input_parameter(self):
        st.markdown(r"Location $\Large{\xi:}$")
        self.loc = st.number_input(label="None", value=0.0, step=0.5, 
                                    label_visibility="collapsed")
        st.markdown(r"Scale $\Large{\omega:}$")
        self.scale = st.number_input(label="None", value=1.0, step=0.5, 
                                     label_visibility="collapsed")
        st.markdown(r"Skewness $\Large{\alpha:}$")
        self.alpha = st.number_input(label="None", value=5.0, step=0.5, 
                                     label_visibility="collapsed")
        self.scipy_obj = scipy.stats.skewnorm(self.alpha, self.loc,
                                              self.scale)
        self.mean = self.scipy_obj.mean()
        self.s_dev = self.scipy_obj.std()

    def validate_parameter(self):
        if self.scale <= 0:
            raise ValueError

    def get_axes(self):
        reach = 4 * self.s_dev
        x_axis = np.linspace(self.mean - reach, self.mean + reach, 70)
        y_axis = self.scipy_obj.pdf(x_axis)
        return x_axis, y_axis
        
    def show_function(self):
        st.latex(r"""\displaystyle{f(x;\xi,\omega,\alpha)=\frac {2}{\omega {\sqrt {2\pi }}}}e^
                 {-{\frac {(x-\xi )^{2}}{2\omega ^{2}}}}\int _{-\infty }^
                 {\alpha \left({\frac {x-\xi }{\omega }}\right)}{\frac{1}
                 {\sqrt {2\pi }}}e^{-{\frac {t^{2}}{2}}}\ dt""")

    def get_sample(self, n):
        return self.scipy_obj.rvs(size=n)


class Normal():

    def input_parameter(self):
        st.markdown(r"$\Large{\mu:}$")
        self.mean = st.number_input(label="None", value=0.0, step=0.5, 
                                    label_visibility="collapsed")
        st.markdown(r"$\Large{\sigma:}$")
        self.s_dev = st.number_input(label="None", value=1.0, step=0.5, 
                                     label_visibility="collapsed")
    
    def validate_parameter(self):
        if self.s_dev <= 0:
            raise ValueError
        
    def get_axes(self):
        reach = 3*self.s_dev
        x_axis = np.linspace(self.mean - reach, self.mean + reach, 40)
        y_axis = scipy.stats.norm.pdf(x_axis, self.mean, self.s_dev)
        return x_axis, y_axis
        
    def show_function(self):
        st.latex(r"""f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}e^{\large{
                 { -\left(\frac{x-\mu}{2\sigma}\right)^{\!2}}}}""")

    def get_sample(self, n):
        return np.random.normal(self.mean, self.s_dev, size=n)


def simulation(obj, n):
    prog_bar = st.progress(0)
    p_values = []
    with st.spinner('Please wait, the simulation is running...'):
        for i in range(1, 1000001):
            if i%100000 == 0:
                prog_bar.progress(int(i/10000))
            sample_data = obj.get_sample(n)
            p_result = scipy.stats.ttest_1samp(sample_data, obj.mean).pvalue
            p_values.append(p_result)
    st.success('Done!')
    return p_values

def show_ava_dist():
    image = Image.open("./test.png")
    st.subheader("Distributions to choose from. Pick one from the left bar")
    st.image(image)

def get_dist_object(dist_choice):
    if dist_choice == "Normal":
        return Normal()
    elif dist_choice == "Skew Normal":
        return SkewNormal()
    elif dist_choice == "Uniform":
        return Uniform()
    elif dist_choice == "Exponential":
        return Exponential()
    elif dist_choice =="Beta":
        return Beta()

def show_graph(obj, dist_choice):
    x_axis, y_axis = obj.get_axes()
    fig, ax = plt.subplots()
    ax.fill_between(x_axis, 0, y_axis, color="maroon")
    ax.set_title(f"Graph of your {dist_choice} Distribution", 
                 fontdict={'fontsize': 18})
    st.pyplot(fig)

def get_fpr(p_values):
    n_sig = len([i for i in p_values if i<0.05])/10000
    return n_sig

def show_pgraph(p_values):
    fig, ax = plt.subplots()
    ax.hist(p_values, bins=40, color="maroon", density=True)
    ax.set_title("The distribution of p-values", fontdict={'fontsize': 18})
    st.pyplot(fig)
        
def callback_button_1 (dist_choice):
    st.session_state[f"button_{dist_choice}"] = True

def update_history(session, dist, n, FPR):
    session["distribution"].append(dist)
    session["sample_size"].append(n)
    session["fpr"].append(FPR)
    session["counter"] += 1

def get_history(session):
    df_index = range(1, len(session["fpr"])+1)
    df = pd.DataFrame({"Distribution": session["distribution"], 
                      "N": session["sample_size"], "FPR": session["fpr"]}, 
                      index=df_index)
    return df

def show_explanation():
    st.subheader("P value and False Positive Rate (FPR)")
    st.write("The one sample t-test will produce a p value. By definition, p value is the "
             "probability of obtaining at least as extreme as a statistic given $h_{0}$ is true. "
             "If the p value of a statistic is below 5%, that is when it is in the critical "
             "region, then the null hypothesis can be rejected. If all assumptions are met, "
             "the FPR, the probability that the t test falsely produce "
             "significant p value when the h0 is actually true, should equal the probability "
             "of obtaining result inside the critical region. Therefore, under the correct use " 
             "of t-test, given the $h_{0}$, the FPR should be 5%. ")
    
    st.subheader("Normality Assumption and FPR")
    st.write("This simulation will satisfy all but the normality assumption. If normality is "
             "violated, the FPR can still be 5% provided that the sample size is large. "
             "However, how large that sample size should be varies depending on the distribution. "
             "The simulation lets you choose between 6 different distributions for the data, "
             "specify their parameters, play with the sample size, and see the resulting FPR.")

    st.subheader("The Monte Carlo Simulation")
    st.write("This program will take 1,000,000 independent samples from a chosen distribution with "
             "a chosen sample size and then perform one sample t test on them with the $h_{0}$ "
             "being true, meaning the value to test the difference from is the true mean of the " 
             "distribution. It will then divide the number of significant p values by 1,000,000."
             "This last quantity is, by definition, the estimated FPR")
    
    st.subheader("Purpose")
    st.write("This program has zero practical benefit :stuck_out_tongue_closed_eyes:. "
             "hopefully you can get a good sense of the normality assumption and an "
             "intuition for the robustness, the degree to which its assumption can be "
             "violated without sacrificing accuracy, of one sample t test.")
    


def main():
    # Initiating the necessary session state
    if "counter" not in st.session_state:
        st.session_state["counter"] = 1
        st.session_state["distribution"] =[]
        st.session_state["sample_size"] = []
        st.session_state["fpr"] = []    
    
    # The navigation sidebar
    with st.sidebar:    
        dist_choice = st.radio("Choose the distribution", 
                               ("Main Menu", "Normal", "Skew Normal", 
                                "Uniform", "Exponential", "Beta"))
        df = get_history(st.session_state)
        st.write("Past Simulations")
        with st.container():
            st.dataframe(df, use_container_width=True)

    # The main menu page
    if dist_choice == "Main Menu":
        st.title("Monte Carlo Simulation to estimate True False Positive Rate of "
                 "one sample T-test When the Normality Assumption is Violated")
        with st.expander("See Explanation"):
            show_explanation()
        show_ava_dist()
    
    # The distributions and simulations page
    else:
        if (f"button_{dist_choice}") not in st.session_state:
            st.session_state[f"button_{dist_choice}"] = False
        dist_object = get_dist_object(dist_choice)
        st.subheader(f"{dist_choice} Distribution")
        st.write("You can adjust the parameters.")
        dist_object.show_function()

        col1, col2 = st.columns(2)
        with col1:
            dist_object.input_parameter()
        try:
            dist_object.validate_parameter()
        except ValueError:
            st.error("Invalid parameters")
        else:
            button_1 = st.button("NEXT", on_click=callback_button_1,
                                    args=(dist_choice,))
            if button_1 or st.session_state[f"button_{dist_choice}"]:
                with col2:
                    show_graph(dist_object, dist_choice)
                sample_size = st.number_input("Sample size: ", value=5, 
                                            min_value=2)
            
                # Hypotheses
                st.write("The hypotheses to be tested")
                st.markdown(rf"$H_{0}:\mu =$ {dist_object.mean}")
                st.markdown(rf"$H_{1}:\mu \neq$ {dist_object.mean}")

                button_b = st.button("SIMULATE") 
                if button_b:
                    p_values = simulation(dist_object, sample_size)
                    fpr = get_fpr(p_values)
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("True FPR: ", fpr)
                    with col4:
                        show_pgraph(p_values)
                    update_history(st.session_state, dist_choice, 
                                sample_size, fpr)                                               


if __name__ =="__main__":
    main()
