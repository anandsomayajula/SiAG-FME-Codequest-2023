# SiAG-FME-Codequest-2023

## **Project Overview**

This project leverages gradient descent optimization to optimize the weights of liquidity pools in a decentralized finance (DeFi) ecosystem. The goal is to enhance the performance of liquidity provision while minimizing the Expected Shortfall (Conditional Value at Risk, CVaR), thereby maximizing returns for liquidity providers and minimizing risk.


## **Objectives and Functionality**

The project includes several key methods and functions to manage liquidity pools and optimize asset allocation. Below are the core functionalities implemented:

### **1. Swap Method (`swap`)**

- Used for token swapping within the liquidity pools.
- If `quote = False`, calculates the number of coins to be received given a gas fee `ϕ`.
- Updates the existing number of coins for assets X and Y.

### **2. Mint Method (`mint`)**

- Mints liquidity pool (LP) tokens for the trader.
- Updates the number of LP tokens held and the reserves of coins X and Y to ensure that the ratio \( \frac{R_x}{R_y} \) remains constant.

### **3. Burn Method (`burn`)**

- Converts LP tokens back into coins X and Y.
- Updates the reserves of each coin and reduces the number of tokens held by the trader by the amount burned.

### **4. Swap and Mint Method (`swap_and_mint`)**

- Combines the `swap` and `mint` operations.
- Determines the amount of tokens that can be minted based on existing ratios and performs the swap followed by the minting process.

### **5. Burn and Swap Method (`burn_and_swap`)**

- Combines the `burn` and `swap` operations.
- Burns the available LP tokens and then swaps the coins based on the best available ratio of Y to X.

## **Optimization Functions**

### **1. Objective Function (`objective(xs_0)`)**

- Evaluates the performance of a given asset allocation (`xs_0`) by simulating trades in the liquidity pools.
- Computes the Conditional Value at Risk (CVaR) and average performance based on simulated trading paths.

### **2. Derivatives Function (`derivatives(xs_0)`)**

- Calculates the numerical gradients of the CVaR to average performance ratio with respect to each asset allocation.
- Utilizes finite difference methods to compute the gradients.

### **3. Gradient Descent Function (`gradientdescent(xs_0)`)**

- Performs gradient descent optimization to find the optimal asset allocation.
- Iteratively updates the allocation in the direction of the negative gradient to minimize CVaR and maximize returns.
- Ensures that allocations remain non-negative and normalized.

## **How to Run the Project**

1. **Initialize the Asset Allocation**:
   - Start with a random asset allocation vector (`xs_0`) normalized to sum to 10.

2. **Run the Gradient Descent Optimization**:
   - Use the `gradientdescent(xs_0)` function to optimize the asset allocation.

3. **View the Results**:
   - The optimized asset allocation and corresponding objective function values (CVaR and performance) will be printed upon completion.

   Example Python script to run the optimization:

   ```python
   from optimization_module import gradientdescent, objective
   
   # Initialize random allocation
   xs_0 = [random values normalized to sum to 10]
   
   # Perform optimization
   optimized_allocation = gradientdescent(xs_0)
   
   # Output results
   print("Optimized Asset Allocation:", optimized_allocation)
   print("Objective Function Values:", objective(optimized_allocation))
   ```

## **Potential Improvements**

- **Fine-tuning hyperparameters**: Adjust learning rates and tolerances to enhance the convergence speed and stability of the optimization algorithm.
- **Sensitivity Analysis**: Perform a sensitivity analysis to understand how parameter changes affect the optimization results.
- **Incorporating Constraints**: Include constraints on asset allocations to align with practical considerations or regulatory requirements.
