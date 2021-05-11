#!/opt/splunk/bin/python

import sys
import os

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.data import path as nltk_data_path
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
CORPORA_DIR = os.path.join(BASE_DIR,'nltk_data')
nltk_data_path.append(CORPORA_DIR)


@Configuration(local=True)
class Vader(StreamingCommand):
    """ Returns sentiment score between -1 and 1, can also return detailed sentiment values.

    ##Syntax

    .. code-block::
        vader textfield=<field>

    ##Description

    Sentiment analysis using Valence Aware Dictionary and sEntiment Reasoner
    Using option full_output will return scores for neutral, positive, and negative which
    are the scores that make up the compound score (that is just returned as the field
    "sentiment". Best to feed in uncleaned data as it takes into account capitalization
    and punctuation.

    ##Example

    .. code-block::
        * | vader textfield=sentence
    """
    text_weight = Option(
        default=50,
        require=False,
        doc='''**Syntax:** **text_weight=***<int>*
        **Description:** The weight of text when analysing combined image and text.''',
        validate=validators.Integer()
        )
    seletced_option = Option(
        default=50,
        doc='''**Syntax:** **text_weight=***<int>*
        **Description:** text/image or both.''',
        ) 


    def stream(self, records):
        sentiment_analyzer = SentimentIntensityAnalyzer()
        for record in records:
            if self.seletced_option == 'get_text' or self.seletced_option == 'get_text_image':
                polarity = sentiment_analyzer.polarity_scores(record['text'])
                text_score = polarity['compound']
                record['sentiment_text'] = text_score
                record['sentiment_text_neutral'] = polarity['neu']
                record['sentiment_text_negative'] = polarity['neg']
                record['sentiment_text_positive'] = polarity['pos']
                record['sentiment'] = text_score

            if self.seletced_option== 'get_image' or self.seletced_option== 'get_text_image':
                polarity_image = sentiment_analyzer.polarity_scores(record['image'])
                image_score = polarity_image['compound']
                record['sentiment_image'] = image_score
                record['sentiment_image_neutral'] = polarity_image['neu']
                record['sentiment_image_negative'] = polarity_image['neg']
                record['sentiment_image_positive'] = polarity_image['pos']
                record['sentiment'] = image_score
            if self.seletced_option == 'get_text_image':
                score_combined = (self.text_weight / 100.0 )* text_score + ((100.0 - self.text_weight) / 100.0 )* image_score
                record['sentiment'] = score_combined
            yield record

dispatch(Vader, sys.argv, sys.stdin, sys.stdout, __name__)
