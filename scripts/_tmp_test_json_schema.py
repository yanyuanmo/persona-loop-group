import os, traceback
os.environ.pop('OPENAI_BASE_URL', None)
from openai import OpenAI
c = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
schema = {
    'name': 'persona_facts', 'strict': True,
    'schema': {
        'type': 'object',
        'properties': {
            'facts': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'slot': {'type': 'string'}, 'value': {'type': 'string'},
                        'dia_id': {'type': 'string'}, 'confidence': {'type': 'number'},
                    },
                    'required': ['slot','value','dia_id','confidence'],
                    'additionalProperties': False,
                },
            }
        },
        'required': ['facts'],
        'additionalProperties': False,
    },
}
for model in ['gpt-4o-mini', 'gpt-4o']:
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{'role':'user','content':'Extract persona facts: user said they like hiking. Return JSON.'}],
            temperature=0, max_tokens=512,
            response_format={'type':'json_schema','json_schema':schema},
        )
        print(f'{model} OK: {r.choices[0].message.content[:200]}')
    except Exception as e:
        traceback.print_exc()
        print(f'{model} ERROR: {e}')
