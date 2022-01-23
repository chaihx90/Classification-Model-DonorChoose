# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 09:35:36 2021

@author: HX
"""

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

projects = pd.read_csv('projects.csv')
outcomes = pd.read_csv('outcomes.csv')

## make boolean labels
outcomes.replace({'f':False, 't': True}, inplace = True)

## Count number of projects by year
projects['year_posted'] = pd.DatetimeIndex(projects.date_posted).year
projects_by_years = projects.groupby(by = 'year_posted').count()['projectid'][:(-1)]
projects_by_years.plot()
plt.xlabel('Year')
plt.ylabel('Number of projects')
plt.title('Number of projects vs Year')
plt.savefig('num_projects_by_years.png', dpi = 600)
plt.show()

## removing projects prior to 2010
min_date = '2010-01-01'
projects = projects.loc[projects.date_posted >= min_date,:]

## Adding the outcome information to the projects, 
## We are not interested in projects without a known outcome at this points
projects_outcome = pd.merge(projects, outcomes.loc[:, ['projectid', 'is_exciting',\
                                                       'at_least_1_teacher_referred_donor',\
                                                       'fully_funded', 'at_least_1_green_donation', 'great_chat',\
                                                       'three_or_more_non_teacher_referred_donors',\
                                                       'one_non_teacher_referred_donor_giving_100_plus',\
                                                       'donation_from_thoughtful_donor']], on = 'projectid')
projects_outcome.date_posted = pd.to_datetime(projects_outcome.date_posted).dt.date

def handle_missing_values(df, recipe):
    """
        Impute the missing values in the df based on the strategy given in
        the recipe dict. The startegies, are kwargs input to sklearn's SimpleImputer, 
        inlcuding a `strategy` and possibly a `fill_value`.
    """
    for column, strategy in recipe.items():
        imp = SimpleImputer(**strategy)
        df.loc[:,column] = imp.fit_transform(df[column].to_numpy().reshape(-1, 1))
    return df

recipe = {'at_least_1_teacher_referred_donor': {'strategy': 'most_frequent'},
          'at_least_1_green_donation': {'strategy': 'most_frequent'},
          'three_or_more_non_teacher_referred_donors': {'strategy': 'most_frequent'},
          'one_non_teacher_referred_donor_giving_100_plus': {'strategy': 'most_frequent'},
          'donation_from_thoughtful_donor': {'strategy': 'most_frequent'}}
projects_outcome = handle_missing_values(projects_outcome, recipe)

projects_outcome['one_or_more'] = projects_outcome['three_or_more_non_teacher_referred_donors'] | \
                                  projects_outcome['one_non_teacher_referred_donor_giving_100_plus'] | \
                                  projects_outcome['donation_from_thoughtful_donor']

req = ['is_exciting', 'at_least_1_teacher_referred_donor', 'fully_funded',\
       'at_least_1_green_donation', 'great_chat', 'one_or_more']

proportion_true = pd.DataFrame(columns = ['Requirement', 'True proportion'])

for i in req:
    proportion_true = proportion_true.append({'Requirement':i, 'True proportion': projects_outcome[i].mean()}, ignore_index = True)
proportion_true.plot.barh(y = 'True proportion', x = 'Requirement')
plt.show()

X = projects_outcome.drop(columns = ['school_ncesid', 'school_metro', 'secondary_focus_subject', 'secondary_focus_area']).copy()
num_ommitted = X.isnull().values.ravel().sum()

if num_ommitted:
    print(f'Dropping {num_ommitted} records out of {len(projects_outcome)} due to missing values')
X.dropna(inplace = True)

## Replace NaN in students_reached with 1 student reached
X['students_reached'] = X['students_reached'].fillna(1)
#bool_var = ['school_charter', 'school_magnet',
#           'school_year_round', 'school_nlns', 'school_kipp',
#           'school_charter_ready_promise',
#           'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
#           'eligible_double_your_impact_match', 'eligible_almost_home_match']
#t_f = {'f':False, 't': True}
#for i in bool_var:
#    projects_outcome[i] = projects_outcome[i].map(t_f)
#true_percent = projects_outcome[bool_var].sum()/projects_outcome[bool_var].count()
#true_percent_df = pd.DataFrame({'boolean variable': true_percent.index, 'Proportion of True': true_percent.values})
#true_percent_df = true_percent_df.sort_values(by = 'Proportion of True', ascending = False)
#sns.barplot(data = true_percent_df, x = 'Proportion of True', y = 'boolean variable')
#plt.show()

#projects_outcome = projects_outcome.drop(columns = ['school_charter', 'school_magnet',
#                                                    'eligible_almost_home_match',
#                                                    'teacher_teach_for_america',
#                                                    'school_year_round',
#                                                    'school_nlns', 'school_kipp',
#                                                    'teacher_ny_teaching_fellow',
#                                                    'school_charter_ready_promise'])
rel_var = ['school_latitude',
           'school_longitude', 'school_city', 'school_state',
           'school_district', 'school_county',
           'teacher_prefix',
           'primary_focus_subject', 'primary_focus_area', 'resource_type',
           'poverty_level', 'grade_level', 'fulfillment_labor_materials',
           'total_price_excluding_optional_support',
           'total_price_including_optional_support', 'students_reached',
           'eligible_double_your_impact_match',
           'school_charter', 'school_magnet',
           'school_year_round', 'school_nlns', 'school_kipp',
           'school_charter_ready_promise',
           'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
           'eligible_almost_home_match']
y = X.is_exciting.copy()
X = X.loc[:,rel_var].copy()


## identifying and label encoding the discrete features 
discrete_cols_mask = X.dtypes.isin(['category', 'bool', np.dtype('O')])
discrete_cols = X.columns[discrete_cols_mask]
X[discrete_cols] = X[discrete_cols].apply(LabelEncoder().fit_transform)

## computing the MI score between all variables and the target
mi_score = MIC(X,y, discrete_features= discrete_cols_mask)

## putting it in a dataframe for convenience
mi_df = pd.DataFrame({'feature': rel_var, 'mutual_information': mi_score})
mi_df.sort_values('mutual_information', ascending=False, inplace=True)
mi_df

sns.set(font_scale = 1.9, rc={'figure.figsize':(11.7,12)})
sns.barplot(data = mi_df, x = 'mutual_information', y = 'feature')
plt.savefig('MI_orig.png', dpi = 600)
plt.show()
### School location and fulfilment_labor_materials
### Need to improve on other indicators such as price, subject areas

#bool_var = ['school_charter', 'school_magnet',
#           'school_year_round', 'school_nlns', 'school_kipp',
#           'school_charter_ready_promise',
#           'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
#           'eligible_double_your_impact_match', 'eligible_almost_home_match']
#t_f = {'f':False, 't': True}
#for i in bool_var:
#    projects_outcome[i] = projects_outcome[i].map(t_f)
#true_percent = projects_outcome[bool_var].sum()/projects_outcome[bool_var].count()
#true_percent_df = pd.DataFrame({'boolean variable': true_percent.index, 'Proportion of True': true_percent.values})
#true_percent_df = true_percent_df.sort_values(by = 'Proportion of True', ascending = False)
#sns.barplot(data = true_percent_df, x = 'Proportion of True', y = 'boolean variable')
#plt.show()

###################################################################
### After proposal ###

#recipe = {'at_least_1_teacher_referred_donor': {'strategy': 'most_frequent'},
#          'at_least_1_green_donation': {'strategy': 'most_frequent'},
#          'three_or_more_non_teacher_referred_donors': {'strategy': 'most_frequent'},
#          'one_non_teacher_referred_donor_giving_100_plus': {'strategy': 'most_frequent'},
#          'donation_from_thoughtful_donor': {'strategy': 'most_frequent'}}
#projects_outcome = handle_missing_values(projects_outcome, recipe)

#projects_outcome['one_or_more'] = projects_outcome['three_or_more_non_teacher_referred_donors'] | \
#                                  projects_outcome['one_non_teacher_referred_donor_giving_100_plus'] | \
#                                  projects_outcome['donation_from_thoughtful_donor']


class DataPrep:
    ## random seed, set for reproducibility
    seed = 424242
    
    ## the main project dataset
    projects_outcome = projects_outcome 
    
    ## frequently referred to columns
    id_column = 'projectid'
    label_column = 'is_exciting'
    
    ## the project IDs in the train and dev set
    ids_split = {}
    
    ## holding features generated per entity, e.g., school features
    entity_datasets = {}

def make_data_splits(dev_share = .2, shuffle = True, recent = False):
    '''
        Split the project IDs into train and dev, then 
        store the result in DataPrep.ids_split
    '''
    projects = DataPrep.projects_outcome
    projects.sort_values(by = 'date_posted', inplace = True)
    train_ids, dev_ids = train_test_split(projects[DataPrep.id_column], test_size= dev_share, shuffle = shuffle, random_state=DataPrep.seed)
    DataPrep.ids_split = {'train': train_ids,'dev': dev_ids, 'all': projects[DataPrep.id_column]}
    if recent:
        ids_df = pd.DataFrame({DataPrep.id_column: train_ids})
        projects_train = pd.merge(ids_df, projects, on = DataPrep.id_column)
        projects = projects_train.loc[projects_train.date_posted >= pd.to_datetime('2012-07-01'),:]
        DataPrep.ids_split['train'] = projects[DataPrep.id_column]
    return train_ids, dev_ids

def downsize_categorical_feature(values, desired_share = .9, max_items = 10):
    ''' Take a set of values of a categorical variable and return the transformed values,
        where value x is transformed to 'other', unless when the items are sorted form the
        most frequent to the least frequent:
        (1) x is aomng the top-k items, with k = max_items
        (2) x is among the top items that cummulatively cover the desiered_share of the data
    '''
    item_counts = values.value_counts()
    cum_share = item_counts.cumsum()/item_counts.sum()
    top_k = cum_share[cum_share < desired_share]
    top_k = top_k[:max_items]
    top_vals = top_k.index
    new_values = values.apply(lambda x : x if x in top_vals else 'Other')
    return new_values

def downsize_all_categorical_features(df, desired_share = .9, max_items = 10,
                                      skip_columns = [DataPrep.id_column, 'schoolid', 'school_ncesid', 'teacher_acctid']):
    '''
        Downsize all categorical columns, except skip_columns, in a dataframe.
    '''
    categorical_columns = df.select_dtypes(['object', 'category']).columns.values.tolist()
    for column in [c for c in categorical_columns if c not in skip_columns]:
        df[column] = downsize_categorical_feature(df[column], desired_share = desired_share, max_items = max_items)
    return df

def dummify_categorical_features(df, exclude_columns = [DataPrep.id_column, 'date_posted']):
    """ Dummify all categorical features (one-hot encoding), 
        except exclud_columns, and return the transformed data frame.
    """
    categorical_columns = df.select_dtypes(exclude = [np.number, np.bool_]).columns.values.tolist()
    categorical_columns = [c for c in categorical_columns if c not in exclude_columns]
    new_df = pd.get_dummies(df, columns=categorical_columns)
    return new_df                                  

def simple_smoothing(num_obs, sum_obs, num_assumed, sum_assumed):
    """
        Compute a weighted avg of the observed and assumed Sum Totals, 
        weighed by the observd and assumed number of observations.
    """
    return (sum_obs + sum_assumed)/(num_obs+num_assumed)

def entity_overall_performance(entity_name, entity_id_column, min_project_num = 3):
    """
        For a given entity, compute historical features and connect these to projects.
        The features include: number of projects, exciting rate and number of projects per year. 
        
        These are only based on the data available in the training set and the ratios are computed
        using a simple_smoothing, and using entities with > min_project_num for computing data-driven priors. 
        
        The output is a dataframe with one row per project, in which missing values are imputed
        using the computed priors.
    """
    projects_outcome = DataPrep.projects_outcome
    ## only use history available in the training data to gather performance information
    train_mask = projects_outcome.projectid.isin(DataPrep.ids_split['train'])
    entity_outcomes = projects_outcome.loc[train_mask,[entity_id_column, DataPrep.id_column, 'date_posted', DataPrep.label_column]].copy()

    ## Compute (naive/absolute) performance of each entity
    entity_performance = entity_outcomes.groupby(entity_id_column).agg(
        num_projects = (DataPrep.id_column, len),
        num_exciting_projects = ( DataPrep.label_column, sum),
        first_project_date = ('date_posted', min),
        last_project_date = ('date_posted', max))
    entity_performance["active_years"] = round((entity_performance.last_project_date - entity_performance.first_project_date).dt.days/365,2)
        
    ## Smoothing: compute data-driven priors. For computing priors, use those with > min_project_num in history.
    more_data_mask = entity_performance.num_projects >= min_project_num
    prior_project_num = entity_performance[more_data_mask].num_projects.mean()
    prior_exciting_project_num = entity_performance[more_data_mask].num_exciting_projects[more_data_mask].mean()
    prior_active_years = entity_performance[more_data_mask].active_years.mean()
    
    ## Smoothing: apply a weighted average on the ratio features.
    entity_performance["pct_exciting"] = simple_smoothing(num_obs = entity_performance.num_projects,
                                                          sum_obs = entity_performance.num_exciting_projects,
                                                          num_assumed = prior_project_num,
                                                          sum_assumed = prior_exciting_project_num)
    entity_performance["projects_per_year"] = simple_smoothing(num_obs = entity_performance.active_years, 
                                                               sum_obs = entity_performance.num_projects,
                                                               num_assumed = prior_active_years,
                                                               sum_assumed = prior_project_num)    
    ## Change column names to be specific to this entity
    entity_performance = entity_performance.add_prefix(entity_name + "_")
    
    ## Join with all the project data (training and test) to attach the features to project_ids
    entity_features = pd.merge(projects_outcome[[DataPrep.id_column, entity_id_column]], entity_performance, 
                               on = entity_id_column, how = 'left')
    
    ## Dealing with missing values (those entitis who were not in the training data)
    missing_recipe = {f'{entity_name}_num_projects': {'strategy': 'constant',
                                                      'fill_value': prior_project_num},
                      f'{entity_name}_pct_exciting': {'strategy': 'constant',
                                                      'fill_value': prior_exciting_project_num/prior_project_num},
                      f'{entity_name}_projects_per_year':{'strategy': 'constant',
                                                          'fill_value': prior_project_num/prior_active_years},
                      f'{entity_name}_active_years': {'strategy': 'constant',
                                                      'fill_value': prior_active_years}
                     }
    handle_missing_values(entity_features, missing_recipe)
    
    ## keeping project_id and the new features
    keep_columns = [DataPrep.id_column, f'{entity_name}_num_projects', f'{entity_name}_pct_exciting', 
                    f'{entity_name}_projects_per_year', f'{entity_name}_active_years']
    entity_features = entity_features.loc[:,keep_columns].copy()
    
    return entity_features  

def entity_past_performance(entity_name, entity_id_column):
    '''
        Compute features that capture an entity's performance, prior to each project's date, 
        using simple forms of smoothing and imputation of missing values.
        The output is a dataframe with one row per project, associating the projectid to the
        entity's previous number of (exciting) projects, and success rate.
    '''
    projects_outcome = DataPrep.projects_outcome.loc[:,[entity_id_column, DataPrep.id_column, 'date_posted', DataPrep.label_column]]
    ## only use history available in the training data to gather performance information
    train_mask = projects_outcome.projectid.isin(DataPrep.ids_split['train'])
    entity_outcomes = projects_outcome.loc[train_mask,:].copy()

    ## gather overall performance, for computing priors
    entity_performance = entity_outcomes.groupby(entity_id_column).agg(
            num_projects = (DataPrep.id_column, len),
            num_exciting_projects = (DataPrep.label_column, sum))
    prior_projects_num = entity_performance.num_projects.mean()
    prior_exciting_projects_num = entity_performance.num_exciting_projects.mean()
    
    ## new column names that will be generated
    num_prev_col = f'{entity_name}_num_prev_projects'
    num_prev_exciting_col = f'{entity_name}_num_prev_exciting_projects'
    pct_prev_exciting_col = f'{entity_name}_pct_prev_exciting'
    
    ## computing past success for each project's entity
    entity_outcomes[num_prev_col] = entity_outcomes.sort_values(by = 'date_posted').groupby(entity_id_column).cumcount()
    entity_outcomes[num_prev_exciting_col] = entity_outcomes.sort_values(by = 'date_posted').groupby(entity_id_column)[DataPrep.label_column].apply(pd.Series.cumsum)
    entity_outcomes[num_prev_exciting_col] = entity_outcomes[num_prev_exciting_col] - entity_outcomes[DataPrep.label_column]
    
    ## smoothing
    entity_outcomes[pct_prev_exciting_col] = simple_smoothing(num_obs=entity_outcomes[num_prev_col],
                                                              sum_obs = entity_outcomes[num_prev_exciting_col],
                                                              num_assumed = prior_projects_num,
                                                              sum_assumed = prior_exciting_projects_num 
                                                             )
        
    entity_features = pd.merge(projects_outcome, entity_outcomes, on = DataPrep.id_column, how = 'left')
    entity_features = entity_features.loc[:,[DataPrep.id_column, num_prev_col, num_prev_exciting_col, pct_prev_exciting_col]] 
    
    ## Missing values: if we haven't seen the entity we simply put 0 in previous performance columns. 
    ## Feel free to change this, e.g., to put the prior counts and percentages.
    entity_features.fillna(0, inplace = True)
    
    return entity_features

def make_project_core_features():
    print('Making project features ...')
    
    ## put here your choice of relevant columns
    use_columns = ['projectid', 'primary_focus_subject', 'primary_focus_area', 'fulfillment_labor_materials',
                    'resource_type', 'date_posted', 'total_price_excluding_optional_support']
    project_features = DataPrep.projects_outcome.loc[:,use_columns].copy()
    
    ## Create your features below:
    
    ## E.g., adding a relative date column
    project_features["date_posted_relative"] = (project_features.date_posted - project_features.date_posted.min()).dt.days

    
    ## E.g., adding a price relative to project's focus area
    price_per_area = project_features.groupby('primary_focus_area')["total_price_excluding_optional_support"].median()
    project_features["area_median_price"] = project_features['primary_focus_area'].map(price_per_area)
    project_features["price_relative_to_focus_area"] = \
        project_features["total_price_excluding_optional_support"]/project_features["area_median_price"]
    
    ## E.g., downsize categorical variables
    downsize_all_categorical_features(project_features, desired_share = .9, max_items = 5)
    
    ## Handling missing values
    missing_recipe = {
                      'primary_focus_area': {'strategy': 'constant', 'fill_value': '_missing_'},
                      'primary_focus_subject': {'strategy': 'constant', 'fill_value': '_missing_'},
                      'resource_type': {'strategy': 'constant', 'fill_value': '_missing_'},
                      'price_relative_to_focus_area' : {'strategy': 'median'}
                     }
    handle_missing_values(project_features, missing_recipe)
    
    ## keep the final columns you'd like to include in the feature set
    project_features.drop(columns = ["date_posted", "area_median_price",
                                     "total_price_excluding_optional_support"],
                          inplace=True)
    
    ## one-hot encoding
    project_features = dummify_categorical_features(project_features)
    
    ## record the features in DataPrep
    DataPrep.entity_datasets['project'] = project_features
    
    return project_features

def make_teacher_features():
    print('Making teacher features ...')
    
    ## put here your choice of relevant columns
    use_columns = ['projectid', 'teacher_acctid', 'date_posted', 'teacher_prefix']    

    teacher_features = DataPrep.projects_outcome.loc[:,use_columns].copy()
    
    ## Create your features below:
    
    ## E.g., adding features related to previous performance of teacher
    teacher_performance = entity_past_performance('teacher', 'teacher_acctid')
    teacher_features = pd.merge(teacher_features, teacher_performance, on = DataPrep.id_column, how = 'left')
    
    ## E.g., downsize categorical variables
    downsize_all_categorical_features(teacher_features, desired_share = .9, max_items = 5)
    
    ## Handling missing values
    missing_recipe = {'teacher_prefix': {'strategy': 'constant', 'fill_value': '_missing_'}}
    handle_missing_values(teacher_features, missing_recipe)
    
    ## keep the final columns you'd like to include in the feature set
    teacher_features.drop(columns = ["date_posted", "teacher_acctid"], inplace=True)
    
    ## one-hot encoding
    teacher_features = dummify_categorical_features(teacher_features)
    
    ## record the features in DataPrep
    DataPrep.entity_datasets['teacher'] = teacher_features
    
    return teacher_features

def make_school_features():
    print('Making school features ...')
    
    ## put here your choice of relevant columns
    use_columns = ['projectid','school_metro']

    school_features = DataPrep.projects_outcome.loc[:,use_columns].copy()
    
    ## Create your features below:
    
    ## E.g., adding features related to overall performance of teachers
    school_performance = entity_overall_performance('school', 'schoolid')
    school_features = pd.merge(school_features, school_performance, on = DataPrep.id_column, how = 'left')
    
    ## E.g., downsize categorical variables, if needed
    downsize_all_categorical_features(school_features, desired_share = .9, max_items = 5)
    
    ## Handling missing values
    missing_recipe = {'school_metro': {'strategy': 'constant', 'fill_value': '_missing_'}}
    handle_missing_values(school_features, missing_recipe)

    ## keep the final columns you'd like to include in the feature set
    school_features.drop(columns = ['school_num_projects'], inplace=True)
    
    ## one-hot encoding
    school_features = dummify_categorical_features(school_features)
    
    ## record the features in DataPrep
    DataPrep.entity_datasets['school'] = school_features
    
    return school_features

def make_geo_features(num_geo_bins = 5):
    print('Making geo features ...')
    
    ## put here your choice of relevant columns
    use_columns = ['projectid','school_latitude', 'school_longitude',
                   'school_city', 'school_state', 'schoolid']

    geo_features = DataPrep.projects_outcome.loc[:,use_columns].copy()
    
    ## Create your features below:
    
    ## E.g., cut lat, long into buckets to capture west/east and north/south-ness
    geo_features["school_latitude_group"] = pd.cut(geo_features.school_latitude, num_geo_bins)
    geo_features["school_longitude_group"] = pd.cut(geo_features.school_longitude, num_geo_bins)
    
    ## E.g., making a city-state combo column to avoid confusing city names in different states
    geo_features["school_city_state"] = geo_features.school_city + "-" + geo_features.school_state
    
    ## E.g., adding features related to size of the city-state in our data
    train_mask = geo_features[DataPrep.id_column].isin(DataPrep.ids_split['train'])
    geo_num_schools = geo_features[train_mask].groupby('school_city_state')\
                        .agg(city_state_num_schools = ('schoolid', pd.Series.nunique),
                             city_state_num_projects = (DataPrep.id_column, pd.Series.nunique))
    geo_num_schools['city_state_projects_per_school'] = geo_num_schools["city_state_num_projects"]/geo_num_schools["city_state_num_schools"]
    geo_features = pd.merge(geo_features, geo_num_schools, on = 'school_city_state', how = 'left')
    
    
    ## E.g., downsize categorical variables, if needed
    downsize_all_categorical_features(geo_features, desired_share = .9, max_items = 5)
    
    ## Handling missing values, if needed
    missing_recipe = {'city_state_projects_per_school': {'strategy': 'median'}}
    handle_missing_values(geo_features, missing_recipe)
                      
    ## rest of the variables can be simply put to zero
    geo_features.fillna(0, inplace=True)

    ## keep the final columns you'd like to include in the feature set
    keep_columns = [DataPrep.id_column,"school_state", "school_latitude_group", "school_longitude_group",
                    "city_state_num_schools", "city_state_num_projects", "city_state_projects_per_school"]
    geo_features = geo_features.loc[:,keep_columns]
    
    ## one-hot encoding
    geo_features = dummify_categorical_features(geo_features)
    
    ## record the features in DataPrep
    DataPrep.entity_datasets['geo'] = geo_features
    
    return geo_features

def make_all_features():
    '''
        Call the feature generation code for all entities. This results
        in entities_dataset to have a datafram associated to each entity.
    '''
    make_project_core_features()
    make_teacher_features()
    make_school_features()
    make_geo_features()

make_data_splits(dev_share = 10000, shuffle= False, recent = True)
make_all_features()

project_exciting = DataPrep.projects_outcome[['projectid', 'is_exciting']]
entities = ['project', 'teacher', 'school', 'geo']
overall_df = pd.DataFrame(columns = ['feature', 'mutual_information'])

for i in entities:
    X = DataPrep.entity_datasets[i].copy()
    y_temp = pd.merge(X, project_exciting, on = 'projectid')
    y = y_temp['is_exciting']
    
    X = X.drop(['projectid'], axis = 1)
    ## identifying and label encoding the discrete features 
    discrete_cols_mask = X.dtypes.isin(['category', 'bool', np.dtype('O')])
    discrete_cols = X.columns[discrete_cols_mask]
    X[discrete_cols] = X[discrete_cols].apply(LabelEncoder().fit_transform)

    ## computing the MI score between all variables and the target
    mi_score = MIC(X,y, discrete_features= discrete_cols_mask)

    ## putting it in a dataframe for convenience
    mi_df = pd.DataFrame({'feature': X.columns, 'mutual_information': mi_score})
    mi_df.sort_values('mutual_information', ascending=False, inplace=True)
    if overall_df.empty:
        overall_df = mi_df.copy()    
    else:
        overall_df = overall_df.append(mi_df)

overall_df = overall_df.sort_values(by = 'mutual_information', ascending = False)

# Plot top 30 features
sns.set(font_scale = 1.9, rc={'figure.figsize':(11.7,12)})
g = sns.barplot(data = overall_df.head(30), x = 'mutual_information', y = 'feature')
#g = g.set(xlim = (0, 0.05))
plt.savefig('MI_top_30.png', dpi = 600)
plt.show(g)

######################### Preparing train, test data ###################################
def get_dataset_by_entities(entities = ['project', 'teacher', 'school', 'geo'], split = 'train', drop_id = True, recent = False):
    ''' 
        Join the datasets for all the entities into one.
        Return the portion of data related to the specified split (train/dev/all)
    '''
    ids = DataPrep.ids_split.get(split, [])
    if (len(ids) == 0):
        print(f'Unknown Split {split}. It can be one of "train", "dev", or "all"')
        return False
    dataset = pd.DataFrame({DataPrep.id_column: ids})
    for entity in entities:
        entity_data = DataPrep.entity_datasets[entity]
        dataset = pd.merge(dataset, entity_data, on = DataPrep.id_column, how = 'left')
        
    ## drop project_id from the columns
    if drop_id:
        dataset.drop(columns = [DataPrep.id_column],inplace = True)
        
    return dataset

def get_labels(split = 'train', drop_id = True):
    '''Returns the labels associatd to each data split (train/dev/all)'''
    ids = DataPrep.ids_split.get(split, [])
    if (len(ids) == 0):
        print(f'Unknown Split {split}. It can be one of "train", "dev", or "all"')
        return False
    ids_df = pd.DataFrame({DataPrep.id_column: ids})
    labels = pd.merge(ids_df, DataPrep.projects_outcome.loc[:,[DataPrep.id_column, DataPrep.label_column]])
    if drop_id:
        labels = labels.loc[:,DataPrep.label_column].values.tolist()
    return labels

def get_train_test_data(entities, recent = False):
    '''
        Get the joined result of all entities features, divided into
        train and test sets.
    '''
    X_train = get_dataset_by_entities(entities, split = 'train')
    y_train = get_labels(split = 'train')
    X_test = get_dataset_by_entities(entities, split = 'dev')
    y_test = get_labels(split = 'dev')
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_train_test_data(entities)

X_train_sel = X_train[overall_df['feature'].head(26)]
X_test_sel = X_test[overall_df['feature'].head(26)]

##################### Build model ##################################
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import datetime

np.random.seed(0)

class Experiment(object):
    '''
        A helper class for capturing each learning experiment.
        Manages training and evaluation.
    '''
    def __init__(self, name, classifier, entities =[]):
        self.name = name
        self.classifier = classifier
        self.pipe = pipeline.Pipeline([('scale', StandardScaler(with_mean=True)),('clf',self.classifier)])
        self.params = classifier.get_params()
        self.entities = entities
        self.timestamp = datetime.datetime.now()
        self.features = []

    def fit_classifier(self, X_train, y_train):
        self.pipe.fit(X_train, y_train)
        return self.pipe
    
    def pred_proba(self, X_test):
        y_pred = self.pipe.predict_proba(X_test)[:,1]
        return y_pred
    
    def evaluate_classifier(self, X_test, y_test):
        self.test_preds_proba = self.pipe.predict_proba(X_test)[:,1]
        self.test_auc = roc_auc_score(y_test, self.test_preds_proba)
        self.test_preds = self.pipe.predict(X_test)
        #self.tn,self.fp,self.fn,self.tp = confusion_matrix(y_test, self.test_preds).ravel()
        prob = pd.concat([pd.Series(y_test, name = DataPrep.label_column), pd.Series(self.test_preds_proba, name = 'prediction')], axis = 1)
        prob = prob.sort_values(by = 'prediction', ascending = False)
        self.top_10_percentile = prob[DataPrep.label_column].head(int(len(y_test)/10)).mean()
        self.top_20_percentile = prob[DataPrep.label_column].head(int(len(y_test)/5)).mean()
        self.top_30_percentile = prob[DataPrep.label_column].head(int(len(y_test)*3/10)).mean()
        self.random = prob[DataPrep.label_column].mean()
        return self.test_auc, self.top_10_percentile , self.top_20_percentile , self.top_30_percentile, self.random

    def get_param(self, param):
        return self.params.get(param, None)
    def get_df_record(self):
        df_record = {'classifier': self.name, 
                'entities': self.entities, 
                'test_roc_auc': self.test_auc,
                'top_10_percentile': self.top_10_percentile,
                'top_20_percentile': self.top_20_percentile,
                'top_30_percentile': self.top_30_percentile,
                'random': self.random,
                'classifier_pipeline': self.classifier,
                'timestamp' : self.timestamp,
                'parameters' : self.params,
                'features': self.features,
                'label': DataPrep.label_column}
        return df_record
    def __repr__(self):
        return f'Experiment: {self.name} \n\t classifier = {self.classifier},\n\t entities = {self.entities})'

### Creating a group of classifiers to experiment with

## Logistic Regression
log_reg = LogisticRegression(max_iter= 200)

## GBM
gbm = GradientBoostingClassifier(verbose = 0,
                                 n_estimators=250,
                                 n_iter_no_change=4,
                                 min_samples_split= 500)

gbm_more = GradientBoostingClassifier(verbose = 0,
                                 n_estimators=1000,
                                 n_iter_no_change=4,
                                 min_samples_split= 500)

## Neural Nets - Fully connected Feed-forward (FFNNs), with different structures
mlp_3 = MLPClassifier(max_iter= 100,verbose=False, 
                      hidden_layer_sizes = (20, 10, 5,), alpha = .001,
                      early_stopping=True, n_iter_no_change=1)

mlp_4 = MLPClassifier(max_iter= 100,verbose=False, 
                      hidden_layer_sizes = (30, 20, 10, 5,), alpha = .001,
                      early_stopping=True, n_iter_no_change=1)

mlp_5 = MLPClassifier(max_iter= 100,verbose=False, 
                      hidden_layer_sizes = (30, 20, 20, 10, 5,), alpha = .001,
                      early_stopping=True, n_iter_no_change=1)
## Random Forrests
rf = RandomForestClassifier(n_estimators = 250,
                            max_depth = 5,
                            min_samples_split = 500)

rf_deep = RandomForestClassifier(n_estimators = 250,
                            max_depth = 10,
                            min_samples_split = 500)

rf_more = RandomForestClassifier(n_estimators = 1000,
                            max_depth = 5,
                            min_samples_split = 500)

rf_deep_more = RandomForestClassifier(n_estimators = 1000,
                            max_depth = 10,
                            min_samples_split = 500)


classifiers = {'Logistic Regression' : log_reg, 'FFNN (d=3)': mlp_3, \
               'FFNN (d=4)': mlp_4, 'FFNN (d=5)': mlp_5, \
               'Random Forest': rf, 'Random Forest (deep)': rf_deep, \
               'Random Forest (more trees)': rf_more, 'Random Forest (deep, more trees)': rf_deep_more,\
               'Gradient Boost': gbm, 'Gradient Boost (more)': gbm_more}
    
experiments = [Experiment(name=n, classifier=c) \
                   for n,c in classifiers.items()]

 ## A dataframe to be filled with the experiment results
experiment_df = pd.DataFrame(columns=['classifier', 'entities', 
                                      'test_roc_auc','top_10_percentile', 
                                      'top_20_percentile', 'top_30_percentile', 'random',
                                      'timestamp', 'classifier_pipeline', 'label'])

## performing the training and evaluation
#for exp in experiments[:]:
#    try:
#        print('='*80)
#        print(exp)

#        print('\n\t.... Starting training ... ')
#        exp.fit_classifier(X_train_sel, y_train)
#        print('\t.... Evaluating ....')
#        test_auc, _, _, _, _ = exp.evaluate_classifier(X_test_sel, y_test)
#        print(f'\tTest AUC: {test_auc}')
        #print(f'\tTrue Positive: {tp}')
        #print(f'\tFalse Positive: {fp}')
        #print(f'\tFalse Negative: {fn}')
        #print(f'\tTrue Negative: {tn}\n')
#        experiment_df = experiment_df.append(exp.get_df_record(), ignore_index=True)
#    except Exception as e:
#        print(f'>>>>Error:{e}\n>>>>Failed in Experiment. Skipping.')


################################ Different requirements ############################################
# fully funded
labels = ['is_exciting', 'fully_funded', 'at_least_1_teacher_referred_donor', \
          'at_least_1_green_donation', 'great_chat', 'one_or_more']
for i in labels:
    DataPrep.label_column = i
    make_all_features()
    X_train, X_test, y_train, y_test = get_train_test_data(entities)
    X_train_sel = X_train[overall_df['feature'].head(26)]
    X_test_sel = X_test[overall_df['feature'].head(26)]
    
    ## performing the training and evaluation
    for exp in experiments[:]:
        try:
            print('='*80)
            print(exp)
    
            print('\n\t.... Starting training ... ')
            exp.fit_classifier(X_train_sel, y_train)
            print('\t.... Evaluating ....')
            test_auc, _, _, _, _ = exp.evaluate_classifier(X_test_sel, y_test)
            print(f'\tTest AUC: {test_auc}')
            #print(f'\tTrue Positive: {tp}')
            #print(f'\tFalse Positive: {fp}')
            #print(f'\tFalse Negative: {fn}')
            #print(f'\tTrue Negative: {tn}\n')
            experiment_df = experiment_df.append(exp.get_df_record(), ignore_index=True)
        except Exception as e:
            print(f'>>>>Error:{e}\n>>>>Failed in Experiment. Skipping.')

idx = experiment_df.groupby(['label'])['test_roc_auc'].transform(max) == experiment_df['test_roc_auc']
best_clf = experiment_df[idx]
best_clf = best_clf.sort_values(by = 'test_roc_auc', ascending = False)
best_clf[['label', 'test_roc_auc']].plot.barh(x = 'label', y = 'test_roc_auc', legend = False)
plt.xlabel('ROC AUC score')
plt.show()

req_pred = pd.DataFrame()
exp = Experiment(name='log_reg', classifier=log_reg)
for i in labels:
    DataPrep.label_column = i
    make_all_features()
    X_train, X_test, y_train, y_test = get_train_test_data(entities)
    X_train_sel = X_train[overall_df['feature'].head(26)]
    X_test_sel = X_test[overall_df['feature'].head(26)]
    
    ## performing the training and evaluation
    print('='*80)
    print(i)
    
    print('\n\t.... Starting training ... ')
    exp.fit_classifier(X_train_sel, y_train)
    print('\t.... Evaluating ....')
    test_auc, _, _, _, _ = exp.evaluate_classifier(X_test_sel, y_test)
    print(f'\tTest AUC: {test_auc}')
    #print(f'\tTrue Positive: {tp}')
    #print(f'\tFalse Positive: {fp}')
    #print(f'\tFalse Negative: {fn}')
    #print(f'\tTrue Negative: {tn}\n')
    req_pred[i] = exp.pred_proba(X_test_sel)

DataPrep.label_column = 'is_exciting'
make_all_features()
X_train, X_test, y_train, y_test = get_train_test_data(entities)
    
req_pred['ensemble'] = req_pred['fully_funded'] *  req_pred['at_least_1_teacher_referred_donor'] * req_pred['at_least_1_green_donation'] * req_pred['great_chat'] * req_pred['one_or_more']
req_pred['actual'] = y_test
roc_auc_score(y_test, req_pred['ensemble'])

prob = pd.concat([pd.Series(y_test, name = 'is_exciting'), pd.Series(req_pred['ensemble'], name = 'prediction')], axis = 1)
prob = prob.sort_values(by = 'prediction', ascending = False)
print(prob['is_exciting'].head(int(len(y_test)/10)).mean())
print(prob['is_exciting'].head(int(len(y_test)/5)).mean())
print(prob['is_exciting'].head(int(len(y_test)/10*3)).mean())
