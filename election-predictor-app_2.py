from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import ElasticNetCV, ElasticNet
import pandas as pd
import numpy as np
import streamlit as st 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

img = Image.open('title_img.png')
st.image(img,use_column_width=True)

def load_data():
    data = pd.read_csv('./Database/CED_model.csv')
    return data

ced_df = load_data()

#defining the feature that will be excluded for the predictor matrix
feats_to_excl = ['divisionnm','divisionid','stateab','partyab','lncvotes','lncpercentage','alpvotes','alppercentage',
                 'swing','election_year','candidateid','givennm','surname','partynm','enrolment','turnout',
                 'turnoutpercentage','turnoutswing','totalpercentage','closeofrollsenrolment','notebookrolladditions',
                 'notebookrolldeletions','reinstatementspostal','reinstatementsprepoll','reinstatementsabsent',
                 'reinstatementsprovisional','year','ced','ced_state','census_year']

data = ced_df.drop(columns=feats_to_excl)

X = data[[col for col in data.columns if 'is_right' not in col]]
y = data['is_right']

baseline = y.value_counts(normalize=True)[1]

lr = LogisticRegression(C=0.1,penalty='l2',solver='newton-cg')

#Defining the training X and Y's, all rows which are not election year 2019
X_train = ced_df[ced_df['year'] != 2019][[col for col in ced_df.columns if 'is_right' not in col]]
X_train = X_train.drop(columns=feats_to_excl)
y_train = ced_df[ced_df['year'] != 2019]['is_right']

#Defining the testing X and Y's, all rows which are election year 2019
X_test = ced_df[ced_df['year'] == 2019][[col for col in ced_df.columns if 'is_right' not in col]]
X_test = X_test.drop(columns=feats_to_excl)
y_test = ced_df[ced_df['year'] == 2019]['is_right']

lr_mod = lr.fit(X_train,y_train)

y_pred = lr_mod.predict(X_test)

def probability_table(model,Xtest,ytest,ref_df):
    probas = model.predict_proba(Xtest)
    prob_df = pd.DataFrame(probas)
    prob_df['electorate'] = ref_df[ref_df['year'] == 2019].loc[:,'divisionnm']
    prob_df.rename(columns={0:'is_left_pct',1:'is_right_pct'},inplace=True)

    prob_df['is_left_pct'] = prob_df['is_left_pct'].apply(lambda x: '{0:.4f}'.format(x*100)).apply(float)
    prob_df['is_right_pct'] = prob_df['is_right_pct'].apply(lambda x: '{0:.4f}'.format(x*100)).apply(float)

    ypred = model.predict(Xtest)
    
    prob_df['predicted'] = ypred
    prob_df['predicted'] = prob_df['predicted'].apply(lambda x: 'right' if x==1 else 'left')
    
    prob_df['actual'] = ytest
    prob_df['actual'] = prob_df['actual'].apply(lambda x: 'right' if x==1 else 'left')
    
    return prob_df

probabs = probability_table(lr_mod,X_test,y_test,ref_df=ced_df)

def logit_convert(x):
    odds = np.exp(x)
    p = odds / (1+ odds)
    return p

#Putting the coefficients into a data frame, zipped up with the feature names
c_coefs_lr = pd.DataFrame(dict(zip(X_train.columns,lr_mod.coef_[0])),index=['Value']).T
c_coefs_lr['ABS_Value'] = c_coefs_lr['Value'].apply(abs)
#Applying the logit convert function to get the probabilities
c_coefs_lr['Probability'] = c_coefs_lr['Value'].apply(logit_convert)
c_coefs_lr.sort_values(by='Value',ascending=False)

while True:
    elec_list = list(ced_df[ced_df['election_year'] == 2019].loc[:,'divisionnm'])
    st.header('Please select an Australian Federal Electorate:')
    key = st.selectbox(
        '',
        (elec_list))
    
    record = probabs[probabs['electorate'].apply(lambda x:str(x).lower()) == key.lower()]
    left = c_coefs_lr.sort_values(by='Value',ascending=False).tail(10).index
    right = c_coefs_lr.sort_values(by='Value',ascending=False).head(10).index
    gr_df = ced_df[ced_df['election_year'] == 2019]
    def percentiler(x,col,df=gr_df):
        ranked = df[col].sort_values().reset_index(drop = True)
        ranked_tab = pd.DataFrame({"data": ranked})
        return len(ranked_tab[ranked_tab['data']<= x])/len(df[col])*100
    
    def bar_labeller(ax,spacing=5,lsize=10):
        """Description:
        A function that passes in the matplotlib object and labels the bar. This only works for bar charts, 
        and is meant to generate neat labels at the top of each bar.

        """
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = spacing
            label = "{0:.2f}%".format(y_value)
            va = 'bottom'
            ax.annotate(
                label,
                (x_value,y_value),
                xytext=(0,space),
                textcoords = 'offset points',
                ha='center',
                va=va,
                fontsize=lsize)

    try:
        cells = record.iloc
        if record.iloc[0,0] > record.iloc[0,1]:
            st.header('Prediction:')
            st.subheader('The seat of ' + str(cells[0,2]) + ' was predicted to vote in a left leaning party.')
            st.subheader('The model gave this prediction a ' + str(round(record.iloc[0,0],2)) + '% probability of occurring.')
            st.header('Actual result:')
            st.subheader('The 2019 election saw ' + str(record.iloc[0,2]) + ' voting for a ' + str(record.iloc[0,4]) + ' leaning party.')
            st.header('Insight into why ' + str(cells[0,2]) + ' voted this way:' )
            st.markdown("""
            Through analysing over 200 different demographics features from the Australian Bureau of Statistics, the model has generated the top 10 features that are most likely
            to predict whether or not the seat will be left or right leaning. 

            Below are the top 10 left and right leaning features the model identified:
            """)
            img2 = Image.open('data_img.png')
            st.image(img2,use_column_width=True)
            st.markdown("""
            For example, for a left leaning seat, the greater the proportion of people who take a tram to work as their primary method of transportation would be a better indicator of the seat\'s
            probability of voting in a left leaning party than its proportion of persons who are non-indigenous. 

            By discovering how your chosen electorate compares to other electorates and the average left or right leaning seat for each of the top 10 predictors,
            you can learn more about why it was predicted to vote this way.

            """)

            top_10L = ced_df[(ced_df['divisionnm'] == record.iloc[0,2]) & (ced_df['election_year'] == 2019)][left].T.iloc[:,0]
            top_10L_ind = top_10L.index
            option = st.selectbox(
                'Please select one of the top ten predictors of a left leaning seat.',
                ('1. Tram : Proportion of persons taking Tram as primary transportation',
                '2. Non_indigenous : Proportion of non-indigenous persons',
                '3. Level_of_education_not_stated : Proportion of persons not stating level of education',
                '4. Never_married : Proportion of persons never married',
                '5. Public_administration_and_safety : Proportion of persons in public admin and safety industry',
                '6. Not_Attending : Proportion of persons not attending an educational institution',
                '7. Mining : Proportion of persons in mining industry',
                '8. No_religion : Proportion of persons with no religion',
                '9. Manufacturing : Proportion of persons in manufacturing industry',
                '10. South_eastern_europe : Proportion of persons with South Eastern European ancestry'))
            for x,y in zip(top_10L_ind,top_10L):
                if str(option.lower().split(' ')[1]) == str(x.lower()[:-4]):
                    if str(int(percentiler(y,x)))[-1] in ['4','5','6','7','8','9','0']:
                        longer = 'The electorate of ' + key +  ' was in the ' + str(int(percentiler(y,x))) + 'th percentile for ' + ' '.join(option.lower().split(' ')[3:]) + 'for the 2019 election.'
                        short = str(int(percentiler(y,x))) + 'th'
                    elif str(int(percentiler(y,x)))[-1] == '1':
                        longer = 'The electorate of ' + key + ' was in the ' + str(int(percentiler(y,x))) + 'st percentile for ' + ' '.join(option.lower().split(' ')[3:]) + 'for the 2019 election.'
                        short = str(int(percentiler(y,x))) + 'st'
                    elif str(int(percentiler(y,x)))[-1] == '2':
                        longer = 'The electorate of ' + key + ' was in the ' + str(int(percentiler(y,x))) + 'nd percentile for ' + ' '.join(option.lower().split(' ')[3:]) + 'for the 2019 election.'
                        short = str(int(percentiler(y,x))) + 'nd'
                    elif str(int(percentiler(y,x)))[-1] == '3':
                        longer = 'The electorate of ' + key +  ' was in the ' + str(int(percentiler(y,x))) + 'rd percentile for ' + ' '.join(option.lower().split(' ')[3:]) + 'for the 2019 election.'
                        short = str(int(percentiler(y,x))) + 'rd'
                    
                    st.subheader('How ' + key + ' ranks for ' + ' '.join(option.lower().split(' ')[3:]) + ':')
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x = gr_df[x],name = "All electorates",))
                    fig.add_trace(go.Scatter(
                        y=[0,0],
                        x=[gr_df[gr_df['divisionnm'] == key][x].iloc[0],gr_df[gr_df['divisionnm'] == key][x].iloc[0]],
                        name=key,
                        mode='markers+text',
                        text=[short],
                        textposition='bottom center',
                        marker=dict(
                            size=15,
                            line=dict(width=2,color='DarkSlateGrey')
                        )))
                    fig.update_layout(
                        title='Electorate percentile scoring',xaxis_title = 'Proportion of persons in electorate (%)',
                        yaxis_title='Number of electorates', autosize=False,width=850,height=500,
                            font=dict(
                                family="Courier New, monospace",))
                    st.plotly_chart(fig) 

                    if st.button('Click here to learn more about this chart',key=1):
                        st.markdown(longer + ' This chart ranks ' + key + '\'s ' + ' '.join(option.lower().split(' ')[3:]) + ', compared to every other Australian Federal Electorate in the 2019 federal election.')

                    left_avg = pd.DataFrame(gr_df.groupby('is_right').mean()[left].loc[0])
                    left_avg.rename(columns={0.0:'Left leanng average'},inplace=True)
                    
                    rec_L = gr_df[gr_df['divisionnm'] == record.iloc[0,2]][left].T
                    rec_L.columns = [record.iloc[0,2]]
                    ax = pd.concat([left_avg,rec_L],axis=1)
                    ax_spec = ax[ax.index == x]

                    st.subheader('How ' + key + ' compares to the average for ' + ' '.join(option.lower().split(' ')[3:]) + ':')
                    fig2 = go.Figure(data=[
                        go.Bar(name=key, x=[''],y=[float(ax_spec.iloc[0,1])], 
                            text=str(round(float(ax_spec.iloc[0,1]),1))+'%',textposition='auto'),
                        go.Bar(name='Left leaning average',x=[''],y=[float(ax_spec.iloc[0,0])],
                            text=str(round(float(ax_spec.iloc[0,0]),1))+'%',textposition='auto')
                    ])

                    fig2.update_layout(title='Comparative bar chart to political orientation average',xaxis_title=option.split(' ')[1],
                    yaxis_title='Proportion of electorate (%)',legend_orientation="h", autosize=False,width=850,height=500,
                        font=dict(
                                family='Courier New,monospace',))
                    
                    st.plotly_chart(fig2)

                    if st.button('Click here to learn more about this chart',key=2):
                        st.markdown('This chart compares ' + key + '\'s ' + ' '.join(option.lower().split(' ')[3:]) + ' for the 2019 federal election to the left leaning average (i.e. The average of all electorates that voted for left leaning parties in the 2019 federal election.)')

                    st.subheader('How ' + key + ' has changed over elections for ' + ' '.join(option.lower().split(' ')[3:]) + ':')
                    fig3 = make_subplots(specs=[[{'secondary_y':True}]])

                    fig3.add_trace(
                        go.Scatter(x=ced_df[ced_df['divisionnm'] == key]['election_year'],
                            y=ced_df[ced_df['divisionnm'] == key][x],name=key),secondary_y=False
                    )

                    fig3.add_trace(
                        go.Scatter(x=ced_df[ced_df['is_right'] == 0].groupby('election_year').mean().index,
                            y=ced_df[ced_df['is_right'] == 0].groupby('election_year').mean()[x],name='Left leaning average'),secondary_y=True)

                    fig3.update_yaxes(title_text='Proportion of ' + key +  ' (%)', secondary_y=False)
                    fig3.update_yaxes(title_text='Proportion of left leaning average (%)', secondary_y=True)
                    fig3.update_xaxes(tickvals=[2004,2007,2010,2013,2016,2019])
                    fig3.update_layout(title='How your electorate has changed over elections',legend_orientation="h",
                        autosize=False,width=850,height=500,
                        font=dict(
                                family='Courier New,monospace',))
        
                    st.plotly_chart(fig3)

                    if st.button('Click here to learn more about this chart',key=3):
                        st.markdown('This chart takes ' + key + '\'s ' + ' '.join(option.lower().split(' ')[3:]) + ' and plots how it has changed throughout every Australian federal election since 2004.')
            break
        elif record.iloc[0,0] < record.iloc[0,1]:
            st.header('Prediction:')
            st.subheader('The seat of ' + str(cells[0,2]) + ' was predicted to vote in a right leaning party.')
            st.subheader('The model gave this prediction a ' + str(round(record.iloc[0,1],2)) + '% probability of occurring.')
            st.header('Actual result:')
            st.subheader('The 2019 election saw ' + str(record.iloc[0,2]) + ' voting for a ' + str(record.iloc[0,4]) + ' leaning party.')
            st.header('Insight into why ' + str(cells[0,2]) + ' voted this way:')
            st.markdown("""
            Through analysing over 200 different demographics features from the Australian Bureau of Statistics, the model has generated the top 10 features that are most likely
            to predict whether or not the seat will be left or right leaning. 

            Below are the top 10 left and right leaning features the model identified:
            """)
            img2 = Image.open('data_img.png')
            st.image(img2,use_column_width=True)
            st.markdown("""
            For example, for a right leaning seat, the greater the proportion of persons who work in wholesale trade would be a better indicator of the seat\'s
            probability of voting in a right leaning party than its proportion of persons who work in retail trade. 

            By discovering how your chosen electorate compares to other electorates and the average left or right leaning seat for each of the top 10 predictors,
            you can learn more about why it was predicted to vote this way.

            """)

            top_10R = ced_df[(ced_df['divisionnm'] == record.iloc[0,2]) & (ced_df['election_year'] == 2019)][right].T.iloc[:,0]
            top_10R_ind = top_10R.index
            option = st.selectbox(
                'Please select one of the top ten predictors of a right leaning seat.',
                ('1. Both_not_stated___both_institution_typp_and_full_time_part_time : Proportion of persons who have not stated their educational institution and not stated their full-time/part-time student status',
                '2. Wholesale_trade : Proportion of persons who work in the wholesale trade industry',
                '3. Retail_trade : Proportion of persons who work in the retail trade industry',
                '4. Year_11_or_equivalent : Proportion of persons who are educated up to year 11 or equivalent',
                '5. Married : Proportion of persons who are married',
                '6. Hrswrkd_49_hours_and_over :  Proportion of persons working 49 hours or over per week',
                '7. Rental_hiring_and_real_estate_services : Proportion of persons who are in rental hiring and real estate services industries',
                '8. Western_Europe : Proportion of persons with Western European ancestry',
                '9. Mortgage_3000_3999 : Proportion of persons holding a mortgage with monthly repayments between $3000 and $3999',
                '10. Employer_government_includes_defence_housing_authority : Proportion of persons whose landlords are the Australian Government (includes Defence Housing Authority) '))
            
            for x,y in zip(top_10R_ind,top_10R):
                if str(option.lower().split(' ')[1]) == str(x.lower()[:-4]) or str(option.lower().split(' ')[1]) == str(x.lower()):
                    if str(int(percentiler(y,x)))[-1] in ['4','5','6','7','8','9','0']:
                        longer = 'The electorate of ' + key +  ' was in the ' + str(int(percentiler(y,x))) + 'th percentile for ' + ' '.join(option.lower().split(' ')[3:]) + 'for the 2019 election.'
                        short = str(int(percentiler(y,x))) + 'th'
                    elif str(int(percentiler(y,x)))[-1] == '1':
                        longer = 'The electorate of ' + key + ' was in the ' + str(int(percentiler(y,x))) + 'st percentile for ' + ' '.join(option.lower().split(' ')[3:]) + 'for the 2019 election.'
                        short = str(int(percentiler(y,x))) + 'st'
                    elif str(int(percentiler(y,x)))[-1] == '2':
                        longer = 'The electorate of ' + key + ' was in the ' + str(int(percentiler(y,x))) + 'nd percentile for ' + ' '.join(option.lower().split(' ')[3:]) + 'for the 2019 election.'
                        short = str(int(percentiler(y,x))) + 'nd'
                    elif str(int(percentiler(y,x)))[-1] == '3':
                        longer = 'The electorate of ' + key +  ' was in the ' + str(int(percentiler(y,x))) + 'rd percentile for ' + ' '.join(option.lower().split(' ')[3:]) + 'for the 2019 election.'
                        short = str(int(percentiler(y,x))) + 'rd'

                    st.subheader('How ' + key + ' ranks for ' + ' '.join(option.lower().split(' ')[3:]) + ':')
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x = gr_df[x],name = "All electorates",))
                    fig.add_trace(go.Scatter(
                        y=[0,0],
                        x=[gr_df[gr_df['divisionnm'] == key][x].iloc[0],gr_df[gr_df['divisionnm'] == key][x].iloc[0]],
                        name=key,
                        mode='markers+text',
                        text=[short],
                        textposition='bottom center',
                        marker=dict(
                            size=15,
                            line=dict(width=2,color='DarkSlateGrey')
                        )))
                    fig.update_layout(
                        title='Electorate percentile scoring',xaxis_title = 'Proportion of persons in electorate (%)',
                        yaxis_title='Number of electorates', autosize=False,width=850,height=500,
                            font=dict(
                                family="Courier New, monospace",))
                    st.plotly_chart(fig)                    
            
                    if st.button('Click here to learn more about this chart',key=1):
                        st.markdown(longer + ' This chart ranks ' + key + '\'s ' + ' '.join(option.lower().split(' ')[3:]) + ', compared to every other Australian Federal Electorate in the 2019 federal election.')

                    right_avg = pd.DataFrame(gr_df.groupby('is_right').mean()[right].loc[1])
                    right_avg.rename(columns={1.0:'Right leanng average'},inplace=True)
            
                    rec_R = pd.DataFrame(gr_df[gr_df['divisionnm'] == record.iloc[0,2]][right].mean(),columns=[record.iloc[0,2]])
                    rec_R = gr_df[gr_df['divisionnm'] == record.iloc[0,2]][right].T
                    rec_R.columns = [record.iloc[0,2]]
                    ax = pd.concat([right_avg,rec_R],axis=1)
                    ax_spec = ax[ax.index == x]

                    st.subheader('How ' + key + ' compares to the average for ' + ' '.join(option.lower().split(' ')[3:]) + ':')
                    fig2 = go.Figure(data=[
                        go.Bar(name=key, x=[''],y=[float(ax_spec.iloc[0,1])], 
                            text=str(round(float(ax_spec.iloc[0,1]),1))+'%',textposition='auto'),
                        go.Bar(name='Right leaning average',x=[''],y=[float(ax_spec.iloc[0,0])],
                            text=str(round(float(ax_spec.iloc[0,0]),1))+'%',textposition='auto')
                    ])

                    fig2.update_layout(title='Comparative bar chart to political orientation average',xaxis_title=option.split(' ')[1],
                    yaxis_title='Proportion of electorate (%)',legend_orientation="h", autosize=False,width=850,height=500,
                        font=dict(
                                family='Courier New,monospace',))
                    
                    st.plotly_chart(fig2)

                    if st.button('Click here to learn more about this chart',key=2):
                        st.markdown('This chart compares ' + key + '\'s ' + ' '.join(option.lower().split(' ')[3:]) + ' for the 2019 federal election to the left leaning average (i.e. The average of all electorates that voted for left leaning parties in the 2019 federal election.)')

                    
                    st.subheader('How ' + key + ' has changed over elections for ' + ' '.join(option.lower().split(' ')[3:]) + ':')
                    fig3 = make_subplots(specs=[[{'secondary_y':True}]])
                    fig3.add_trace(
                        go.Scatter(x=ced_df[ced_df['divisionnm'] == key]['election_year'],
                            y=ced_df[ced_df['divisionnm'] == key][x],name=key),secondary_y=False
                    )
                    fig3.add_trace(
                        go.Scatter(x=ced_df[ced_df['is_right'] == 1].groupby('election_year').mean().index,
                            y=ced_df[ced_df['is_right'] == 1].groupby('election_year').mean()[x],name='Right leaning average'),secondary_y=True)

                    fig3.update_yaxes(title_text='Proportion of ' + key +  ' (%)', secondary_y=False)
                    fig3.update_yaxes(title_text='Proportion of right leaning average (%)', secondary_y=True)
                    fig3.update_xaxes(tickvals=[2004,2007,2010,2013,2016,2019])
                    fig3.update_layout(title='How your electorate has changed over elections',legend_orientation="h",
                        autosize=False,width=850,height=500,
                        font=dict(
                                family='Courier New,monospace',))
        
                    st.plotly_chart(fig3)
                    
                    if st.button('Click here to learn more about this chart',key=3):
                        st.markdown('This chart takes ' + key + '\'s ' + ' '.join(option.lower().split(' ')[3:]) + ' and plots how it has changed throughout every Australian federal election since 2004.')

            break
        else:
            print('Seat is equally left and right')
            break
    except IndexError:
        st.write('Electorate not found, please try again.')

st.header('About this model:')
st.markdown("""
For further information regarding various features of how this model was made and implemented please select below:
""")

content = st.selectbox(
    'Contents:',
    ('Objective','Process Overview','Data Sources','Structure of Database','How does it work?', 'Results','Applications'))

img3 = Image.open((content.lower().replace(' ','_') + '.png'))
st.image(img3,use_column_width=True)

st.header('About the author:')
st.markdown("""
Thank you for visitng!

My name is Peter Rudder, I am a recent graduate of Data Science at General Assembly Sydney. I am currently seeking Data Science opportunities in Sydney, Australia. 
This web app is a continuation of my final project that I wrote while at General Assembly. 

If you would like to ask me more questions about this model:
    
    Email: peterwrudder@outlook.com
    Linkedin: linkedin.com/in/peterrudder
    Github: github.com/prudder/
""")