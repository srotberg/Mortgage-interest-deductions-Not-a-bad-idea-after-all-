This file shows you how to run the simulations in "Mortgage interest deductions? Not a bad idea after all", by Rotberg S., and Steinberg J., Journal of Monetary Economics.

Before running each simulation you will have to run the following commands:

Write the command in the line below and press enter:
gfortran MID_not_a_bad_idea_after_all_Rotberg_Steinberg_JME.f90 -I/usr/local/include -lm -mcmodel=medium -fopenmp -g -o MID_not_a_bad_idea_after_all

Write the command in the line below and press enter:
./MID_not_a_bad_idea_after_all

At the very top of the Fortran code there is a module titled "Global_Vars". At the top of this module you will see several variables, which we describe below how to set for each simulation.

1. Baseline

	a. Benchmark

		is_counterfactual=0
        	endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1 
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	b. Repealing MID

		is_counterfactual=1
        	endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=0
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1 
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	c. Taxing imputed rents

		is_counterfactual=1
        	endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=1
        	low_elasticity_of_housing=1
	 	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1 
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

2. Infinite rental supply elasticity
	
	a. Benchmark

		is_counterfactual=0
        	endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=1
		other_rental_supply_elasticity=0
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	b. Repealing MID

		is_counterfactual=1
        	endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=0
        	tax_imputed_rent=1
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=1
		other_rental_supply_elasticity=0
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	c. Taxing imputed rents

		is_counterfactual=1
        	endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=1
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=1
		other_rental_supply_elasticity=0
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

3. No min. rental

	a. Benchmark

		is_counterfactual=0
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=1
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	b. Repealing MID

		is_counterfactual=1
        	endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=0
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	c. Taxing imputed rents

		is_counterfactual=1
        	endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=1
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

Senstivity analyses
1. 50% higher rental elasticity

	a. Benchmark

		is_counterfactual=0
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=1
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	b. Repealing MID

		is_counterfactual=1
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=0
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=1
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	c. Taxing imputed rents

		is_counterfactual=0
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=0
        	tax_imputed_rent=1
        	low_elasticity_of_housing=1
        	high_rental_elasticity=1
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

2. 50% smaller min. rental

	a. Benchmark

		is_counterfactual=0
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=1
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	b. Repealing MID

		is_counterfactual=1
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=0
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=1
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	c. Taxing imputed rents

		is_counterfactual=1
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=1
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=1
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

3. Elastic housing supply

	a. Benchmark

		is_counterfactual=0
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=0
        	low_elasticity_of_housing=0
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	b. Repealing MID

		is_counterfactual=1
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=0
        	tax_imputed_rent=0
        	low_elasticity_of_housing=0
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	c. Taxing imputed rents

		is_counterfactual=1
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=1
        	low_elasticity_of_housing=0
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

4. Endogenous labor supply

	a. Benchmark

		is_counterfactual=0
       		endogenous_rental_market=0
        	include_labor_supply=1
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	b. Repealing MID

		is_counterfactual=1
       		endogenous_rental_market=0
        	include_labor_supply=1
		include_moving_back=0
        	mortgage_deductibility=0
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	c. Taxing imputed rents

		is_counterfactual=1
       		endogenous_rental_market=0
        	include_labor_supply=1
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=1
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

5. Endogenous landlords

	a. Benchmark

		is_counterfactual=0
       		endogenous_rental_market=1
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	b. Repealing MID

		is_counterfactual=1
       		endogenous_rental_market=1
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=0
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	c. Taxing imputed rents

		is_counterfactual=1
       		endogenous_rental_market=1
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=1
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

6. Moving with parents

	a. Benchmark

		is_counterfactual=0
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=1
        	mortgage_deductibility=1
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	b. Repealing MID

		is_counterfactual=1
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=1
        	mortgage_deductibility=0
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

	c. Taxing imputed rents

		is_counterfactual=1
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=1
        	mortgage_deductibility=1
        	tax_imputed_rent=1
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=0
        	rent_voucher=0

7. Rent vouchers instead of labor income tax cuts

	a. Repealing MID

		is_counterfactual=1
       		endogenous_rental_market=1
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=0
        	tax_imputed_rent=0
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=1
        	rent_voucher=1

	b. Taxing imputed rents

		is_counterfactual=1
       		endogenous_rental_market=0
        	include_labor_supply=0
		include_moving_back=0
        	mortgage_deductibility=1
        	tax_imputed_rent=1
        	low_elasticity_of_housing=1
        	high_rental_elasticity=0
        	half_smallest_rental=0
        	smaller_smallest_rental=0
		infinite_elasticity_of_rental_supply=0
		other_rental_supply_elasticity=1
		rent_voucher_to_lowest_incomes=1
        	rent_voucher=1