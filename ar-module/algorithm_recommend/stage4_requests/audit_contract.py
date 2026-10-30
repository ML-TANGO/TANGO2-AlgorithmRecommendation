"""Strict semantic-audit response contract; no model imports required."""
import json

def parse(text,request):
    try:
        a=text.index('{');b=text.rindex('}')+1;x=json.loads(text[a:b])
        if set(x)!={'faithful','issues','thinking_recommended','thinking_reason','evidence'}:return None
        if type(x['faithful']) is not bool or type(x['thinking_recommended']) is not bool:return None
        if not isinstance(x['issues'],list) or not all(isinstance(t,str) for t in x['issues']):return None
        if x['faithful'] and x['issues']:return None
        if not x['faithful'] and not x['issues']:return None
        if not isinstance(x['thinking_reason'],str) or not 5<=len(x['thinking_reason'])<=250:return None
        if x['thinking_recommended']:
            if not isinstance(x['evidence'],str) or len(x['evidence'])<4 or x['evidence'] not in request:return None
        elif x['evidence'] is not None:return None
        return x
    except (ValueError,KeyError,TypeError):return None

