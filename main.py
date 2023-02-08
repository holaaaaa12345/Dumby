import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-dark")
import pandas as pd
from scipy.stats import chi2, beta, expon, uniform, skewnorm, norm, ttest_1samp
from PIL import Image

NUMBER_OF_ITERATIONS = 500_000


# Each distributions have their own class
class ChiSquare():

    def input_parameter(self):
        st.markdown(r"$\Large k:$")
        self.df = st.number_input(label="None", value=1, step=1, 
                                  label_visibility="collapsed")
        self.scipy_object = chi2(df=self.df)
        self.mean = self.scipy_object.mean()
        self.s_dev = self.scipy_object.std()
    
    def validate_parameter(self):
        if self.df <= 0:
            raise ValueError

    def get_axes(self):
        reach = 4 * self.s_dev
        x_axis = np.linspace(0, self.mean + reach, 70)
        y_axis = self.scipy_object.pdf(x_axis)
        return x_axis, y_axis
    
    @staticmethod
    def show_function():
        st.markdown(r"""$f(x; k) = \frac{(1/2)^{k/2}}{\Gamma(k/2)} x^{k/2 - 1} 
                    e^{-x/2}$ where $\Gamma(x) = \int_0^{-\infty} 
                    t^{x - 1} e^{-t} dt$ and $k\in\mathbb{N}^{+}$""")
        
    def get_sample(self, n):
        return np.random.chisquare(df=self.df, size=n)
    

class Beta():

    def input_parameter(self):
        st.markdown(r"$\Large\alpha:$")
        self.alpha = st.number_input(label="None", value=0.5, step=0.5, 
                                     label_visibility="collapsed")
        st.markdown(r"$\Large\beta:$")
        self.beta = st.number_input(label="None", value=0.7, step=0.5, 
                                    label_visibility="collapsed")
        self.scipy_object = beta(a=self.alpha, b=self.beta)
        self.mean = self.scipy_object.mean()
    
    def validate_parameter(self):
        if self.alpha <= 0 or self.beta <=0:
            raise ValueError

    def get_axes(self):
        x_axis = np.linspace(0, 1, 50)
        y_axis = self.scipy_object.pdf(x_axis)
        return x_axis, y_axis
    
    @staticmethod
    def show_function():
        st.markdown(r"""$f(x; \alpha,\beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1}
                    (1 - x)^{\beta - 1}$ where $B(\alpha, \beta) = \int_0^1 t^{\alpha - 1}
                    (1 - t)^{\beta - 1} dt$ and $\alpha>0$ and $\beta>0$""")

    def get_sample(self, n):
        return np.random.beta(a=self.alpha, b=self.beta, size=n)


class Exponential():

    def input_parameter(self):
        st.markdown(r"$\Large\beta:$")
        self.beta = st.number_input(label="None", value=1.0, step=0.5, 
                                    label_visibility="collapsed")
        self.scipy_object = expon(loc=0, scale=self.beta)
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

    @staticmethod
    def show_function():
        st.markdown(r"""$f(x; \frac{1}{\beta}) = \frac{1}{\beta} 
                    e^{-\frac{x}{\beta}}$ where $\beta>0$""")

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
        self.scipy_object = uniform(self.a, self.b-self.a)
        self.mean = self.scipy_object.mean()
        self.s_dev = self.scipy_object.std()
    
    def validate_parameter(self):
        if self.a >= self.b:
            raise ValueError
        
    def get_axes(self):
        x_axis = np.array([self.a, self.b])
        y_axis = self.scipy_object.pdf(x_axis)
        return x_axis, y_axis
        
    @staticmethod
    def show_function():
        st.markdown(r"""$f(x; a,b) = {\begin{cases}{\frac {1}{b-a}}&{\text{for }}x\in [a,b]\\
                    0&{\text{otherwise}}\end{cases}}$ where $a<b$""")

    def get_sample(self, n):
        return np.random.uniform(self.a, self.b, size=n)


class SkewNormal():

    def input_parameter(self):
        st.markdown(r"$\Large{\xi}$ (Location) :")
        self.loc = st.number_input(label="None", value=0.0, step=0.5, 
                                   label_visibility="collapsed")
        st.markdown(r"$\Large{\omega}$ (Scale) :")
        self.scale = st.number_input(label="None", value=1.0, step=0.5, 
                                     label_visibility="collapsed")
        st.markdown(r"$\Large{\alpha}$ (Skewness) :")
        self.alpha = st.number_input(label="None", value=5.0, step=0.5, 
                                     label_visibility="collapsed")
        self.scipy_object = skewnorm(self.alpha, self.loc,
                                                 self.scale)
        self.mean = self.scipy_object.mean()
        self.s_dev = self.scipy_object.std()

    def validate_parameter(self):
        if self.scale <= 0:
            raise ValueError

    def get_axes(self):
        reach = 4 * self.s_dev
        x_axis = np.linspace(self.mean - reach, self.mean + reach, 70)
        y_axis = self.scipy_object.pdf(x_axis)
        return x_axis, y_axis

    @staticmethod    
    def show_function():
        st.markdown(r"""$\displaystyle{f(x;\xi,\omega,\alpha)=\frac {2}{\omega {\sqrt {2\pi }}}}e^
                    {-{\frac {(x-\xi )^{2}}{2\omega ^{2}}}}\int _{-\infty }^
                    {\alpha \left({\frac {x-\xi }{\omega }}\right)}{\frac{1}
                    {\sqrt {2\pi }}}e^{-{\frac {t^{2}}{2}}}\ dt$ where $\omega>0$""")

    def get_sample(self, n):
        return self.scipy_object.rvs(size=n)


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
        y_axis = norm.pdf(x_axis, self.mean, self.s_dev)
        return x_axis, y_axis
    
    @staticmethod
    def show_function():
        st.markdown(r"""$f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}}e^{\large{
                    { -\left(\frac{x-\mu}{2\sigma}\right)^{\!2}}}}$ where $\sigma>0$""")

    def get_sample(self, n):
        return np.random.normal(self.mean, self.s_dev, size=n)


def simulation(obj, n, n_iter):
    prog_bar = st.progress(0)
    p_values = []
    ten_percent = n_iter/10
    denom = n_iter/100
    with st.spinner('Please wait, the simulation is running...'):
        for i in range(1, n_iter+1):
            if i%ten_percent == 0:
                prog_bar.progress(int(i/denom))
            sample_data = obj.get_sample(n)
            p_result = ttest_1samp(sample_data, obj.mean).pvalue
            p_values.append(p_result)
    st.success('Done!')
    return p_values

def show_ava_dist():
    st.subheader("Distributions to choose from. Pick one from the left bar")
    with Image.open("./distributions.png") as dist_image:
        st.image(dist_image)

def get_dist_object(dist_choice):
    if dist_choice == "Normal":
        return Normal()
    elif dist_choice == "Skew Normal":
        return SkewNormal()
    elif dist_choice == "Uniform":
        return Uniform()
    elif dist_choice == "Exponential":
        return Exponential()
    elif dist_choice == "Beta":
        return Beta()
    elif dist_choice == "Chi Square":
        return ChiSquare()

def show_graph(obj, dist_choice):
    x_axis, y_axis = obj.get_axes()
    fig, ax = plt.subplots()
    ax.fill_between(x_axis, 0, y_axis, color="maroon")
    ax.set_title(f"Plot of your {dist_choice} Distribution", 
                 fontdict={'fontsize': 18})
    st.pyplot(fig)

def get_fpr(p_values, n_iter):
    denom = n_iter/100
    n_sig = len([i for i in p_values if i<0.05])/denom
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
    with open("explanation.txt", "r") as file:
        exp = file.read()
    st.markdown(exp, unsafe_allow_html=True)

def main():

    # Initiating the history session state
    if "counter" not in st.session_state:
        st.session_state["counter"] = 1
        st.session_state["distribution"] =[]
        st.session_state["sample_size"] = []
        st.session_state["fpr"] = []    
    
    # The navigation sidebar
    with st.sidebar:
        st.subheader("Navigation")    
        dist_choice = st.radio("None", 
                               ("Main Menu", "Normal", "Skew Normal", 
                               "Uniform", "Exponential", "Beta",
                               "Chi Square"),
                               label_visibility="collapsed")
        df = get_history(st.session_state)
        st.subheader("Simulation History")
        with st.container():
            st.dataframe(df, use_container_width=True)

    # The main menu page
    if dist_choice == "Main Menu":
        st.title("Monte Carlo Simulation to estimate False Positive Rate of "
                 "one sample t-test when the Normality Assumption is violated*")
        with st.expander("See Explanation"):
            show_explanation()
        show_ava_dist()
        st.caption("*Inspired by a youtube video made by jbstatistics "
                   "(https://www.youtube.com/watch?v=U1O4ZFKKD1k&ab_channel=jbstatistics)")
        
    # The distributions and simulations page
    else:
        if (f"button_{dist_choice}") not in st.session_state:
            st.session_state[f"button_{dist_choice}"] = False
        
        dist_object = get_dist_object(dist_choice)
        st.subheader(f"{dist_choice} Distribution")
        st.write("You can adjust the parameters.")
        dist_object.show_function()

        col1, col2 = st.columns(2)

        # Parameter input and validation
        with col1:
            dist_object.input_parameter()
        try:
            dist_object.validate_parameter()
        except ValueError:
            st.error("Invalid parameter(s). Please refer to the stated PDF.")
        
        # Sample size input and plot output
        else:
            button_1 = st.button("NEXT", on_click=callback_button_1,
                                 args=(dist_choice,))
            if button_1 or st.session_state[f"button_{dist_choice}"]:
                with col2:
                    show_graph(dist_object, dist_choice)
                sample_size = st.number_input("Sample size: ", value=5, 
                                              min_value=2)
            
                # Hypothesis
                st.markdown("__The hypothesis to be tested__")
                st.markdown(rf"$H_{0}:\mu =$ {dist_object.mean}")
                st.markdown(rf"$H_{1}:\mu \neq$ {dist_object.mean}")

                button_b = st.button("SIMULATE") 
                if button_b:
                    p_values = simulation(dist_object, sample_size, NUMBER_OF_ITERATIONS)
                    fpr = get_fpr(p_values, NUMBER_OF_ITERATIONS)

                    # Show Results
                    col3, col4 = st.columns(2)
                    with col3:
                        st.metric("FPR: ", fpr)
                    with col4:
                        show_pgraph(p_values)
                    update_history(st.session_state, dist_choice, 
                                   sample_size, fpr)                                               


if __name__ =="__main__":
    main()
