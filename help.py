import pandas
import fancyimpute
import numpy
import matplotlib.pyplot

def describe(config):

    """ Function to describe the modeling data."""
    df = pandas.read_csv(config.raw_file, delimiter=',')
    return df.describe(include='all').transpose()

def feature_create(config):

    """ Function to describe the modeling data."""
    df = pandas.read_csv(config.raw_file, delimiter=',')

    # Feature creation title
    
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
    df.loc[df["Title"] == 'Mr','Title2'] = 'Mr'
    df.loc[df["Title"] == 'Mrs','Title2'] = 'Mrs'
    df.loc[df["Title"] == 'Miss','Title2'] = 'Miss'
    df.loc[df["Title"] == 'Master','Title2'] = 'Master'
    df.loc[df["Title"] == 'Ms','Title2'] = 'Ms'
    df.loc[df["Title"] == 'Mlle','Title2'] = 'Miss'
    df.loc[df["Title"] == 'Ms','Title2'] = 'Miss'
    df.loc[df["Title"] == 'Mme','Title2'] = 'Mrs'
    df.loc[df["Title2"].isna(),'Title2'] = 'Rare'
    df_title = pandas.get_dummies(df.Title2 , prefix = 'Title')
    df = pandas.concat([df,df_title],axis=1)
    df['Name_len'] = df.Name.str.len()
    df['Name_space'] = df.Name.str.count(' ')
    df = df.drop(['Name'], axis=1)
    df = df.drop(['Title'], axis=1)
    df = df.drop(['Title2'], axis=1)

    # Missing Value treatment for Cabin

    df['Cabin_new'] = df.Cabin.str[:1]
    df = df.drop(['Cabin'], axis=1)
    df_cabin = pandas.get_dummies(df.Cabin_new,prefix = 'Cabin', dummy_na = True)
    df = pandas.concat([df,df_cabin],axis=1)
    df = df.drop(['Cabin_new'], axis=1)

    # Feature creation Sex

    df['Gender'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    df_sex = pandas.get_dummies(df.Sex , prefix = 'Sex')
    df = pandas.concat([df,df_sex],axis=1)
    df = df.drop(['Sex'], axis=1)

    # Family

    df['FamilySize'] = df['SibSp'] + df['Parch'] +1
    df['withsomebody'] = df['SibSp'] + df['Parch']
    df["isalone"] = df['withsomebody'].copy()
    df["isalone"].loc[df['withsomebody'] > 0] = 0
    df["isalone"].loc[df['withsomebody'] == 0] = 1

    # Missing Value treatment for Embarked

    df['Embarked'] = df['Embarked'].fillna('C')
    df_Embarked = pandas.get_dummies(df.Embarked , prefix = 'Embarked')
    df = pandas.concat([df,df_Embarked],axis=1)
    df = df.drop(['Embarked'], axis=1)

    # Missing Value treatment for Ticket

    new = df["Ticket"].str.split(" ", n = 2, expand = True)
    new[3] = numpy.where(new[2].isna(),new[1],new[2])
    new['Ticket1'] = numpy.where(new[3].isna(),new[0],new[3])
    new['Ticket2'] = new[0].str.extract('([A-Za-z]+)',expand=False)
    new['T_length'] = new.Ticket1.str.len()
    new['T_First'] = new.Ticket1.str[:1]
    new = new.drop([0], axis=1)
    new = new.drop([1], axis=1)
    new = new.drop([2], axis=1)
    new = new.drop([3], axis=1)
    new = new.drop(['Ticket1'], axis=1)
    df = pandas.concat([df,new],axis=1)
    df = df.drop(['Ticket'], axis=1)

    df.loc[df['T_length'] < 5,'T_l_new'] = 'S'
    df.loc[df['T_length'] == 5,'T_l_new'] = 'M'
    df.loc[df['T_length'] > 5,'T_l_new'] = 'L'
    df.loc[df['T_First'] == '1','T_f_new'] = 'S'
    df.loc[df['T_First'] == '2','T_f_new'] = 'M'
    df.loc[df['T_f_new'].isna(),'T_f_new'] = 'L'
    df['High_ticket'] = df['Ticket2'].isin(['PP','PC','C','P'])
    df_t1 = pandas.get_dummies(df.T_l_new, prefix = 'T_l')
    df = pandas.concat([df,df_t1],axis=1)
    df_t2 = pandas.get_dummies(df.T_f_new, prefix = 'T_F')
    df = pandas.concat([df,df_t2],axis=1)

    df = df.drop(['T_l_new'], axis=1)
    df = df.drop(['T_f_new'], axis=1)
    df = df.drop(['T_First'], axis=1)
    df = df.drop(['T_length'], axis=1)
    df = df.drop(['Ticket2'], axis=1)

    # interaction between class and age

    df['Age*Class'] = df["Age"]*df["Pclass"]

    # interaction between class and child

    df.loc[df['Age'] < 16,'ischild'] = 1
    df.loc[df.ischild.isna(),'ischild'] = 0
    df["Child*Class"] = df["ischild"]*df["Pclass"]

    # interaction between class and gender

    df["Gender*Class"] = df["Gender"]*df["Pclass"]

    # Missing Value treatment for Age

    df_ii = pandas.DataFrame(fancyimpute.IterativeImputer().fit_transform(df.drop('Survived', axis=1)))
    df_ii.columns = df.drop('Survived', axis=1).columns
    df_ii.index = df.index
    df_ii = pandas.concat([df_ii,df.Survived],axis=1)
    df_ii.loc[df_ii['_data_'] == 0,'Survived'] = float('NaN')
    df = df_ii

    df.loc[df['Age'] <= 26,'Age_new'] = 'A'
    df.loc[(df['Age'] > 16) & (df['Age'] <= 26), 'Age_new'] = 'B'
    df.loc[(df['Age'] > 26) & (df['Age'] <= 36), 'Age_new'] = 'C'
    df.loc[(df['Age'] > 36) & (df['Age'] <= 62), 'Age_new'] = 'D'
    df.loc[df['Age'] > 62,'Age_new'] = 'E'

    df_age = pandas.get_dummies(df.Age_new,prefix = 'Age')
    df = pandas.concat([df,df_age],axis=1)
    df = df.drop(['Age_new'], axis=1)

    df.to_csv(config.feature_file, index=False)

    return None

def feature_plot(config):

    """ Function to describe the modeling data."""

    df = pandas.read_csv(config.feature_file, delimiter=',')
    matplotlib.pyplot.hist(df['Age'])
    fig = matplotlib.pyplot.figure()
    fig.savefig('/home/swayush/ML/test.png',bbox_inches='tight')
    matplotlib.pyplot.close(fig)





# todo: Feature understanding
# todo: Missing Value
# todo: Feature engineering
# todo: Feature understanding
# todo: Sampling code