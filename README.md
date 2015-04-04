# ActivityDynamics
Framework for calculating activity dynamics for networks.

# Dependencies
To be able to run the activity dynamics framework the following Python libraries have to be installed:

- NumPy
- SciPy 
- Matplotlib
- Graph-Tool

Other software needed to create result plots:

- R

# Setup
Run setup.py and set r_binary_path in config.py

# Calculate activity dynamics for Zachary's Karate Club (synthetic network with random weights)

*Note that calculations are computationally intensive and may take some time to finish!*
Once all dependencies are installed we can:

<ul>
<li>Run 'python setup.py'</li>
<li>Run 'python calc_dynamics_synthetic.py'</li>
  <ul>
    <li>The process will fork 10 additional processes to speed up the calculation of activity dynamics.</li>
    <li>There is a lot of colorful debug output to watch :).</li>
    <li>The message "-> INFO: Could not store empirical activities!" occurs for all synthetic datasets</li>
  </ul>
<li>Wait until calc_dynamics_synthetic.py has finished</li>
<li>Browse to 'results/graphs/" and look at the generated Karate_* graphs</li>
<li>Browse to 'results/plots/weights_over_time/Karate/' and look at the Karate_ratio_*.pdf files.</li>
</ul>

# FYI

The data_preparation folder contains all the python scripts necessary to process instances of the StackExchange dataset and to some
extent also Semantic MediaWiki instances. Also, due to the dependency on graph-tool, the framework only works on Linux and Mac OSX environments.