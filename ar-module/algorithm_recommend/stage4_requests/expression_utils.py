"""GPU aliases omit capacity while preserving Ti, Ada and workstation distinctions."""
import re


def gpu_identity(gpu):
    return re.sub(r'\s*\d+(?:\.\d+)?\s*G(?:i)?B\b', '', gpu['gpu_name'], flags=re.I).strip()


DIGITS=dict(zip("0123456789","공일이삼사오육칠팔구"))

def gpu_aliases(gpu):
    gpu=dict(gpu)
    gpu['gpu_name']=gpu_identity(gpu)
    gpu['full_name']=re.sub(r'\s*\d+(?:\.\d+)?\s*G(?:i)?B\b','',gpu['full_name'],flags=re.I).strip()
    name=gpu['gpu_name'];variants=[]
    def add(style,text):
        if text not in [v['text'] for v in variants]:variants.append({'style':style,'text':text})
    add('full_name',gpu['full_name']);add('canonical',name)
    add('hyphenated',name.replace(' ','-'));add('compact',name.replace(' ',''))
    add('lowercase',name.lower());add('brand_korean','엔비디아 '+name)
    if name.startswith(('GTX ','RTX ')) and name.split()[1].isdigit() and name.split()[1] not in {'5000','6000'}:
        number=name.split()[1];suffix=' Ti' if 'Ti' in name else ''
        add('colloquial_number',number+suffix+' 그래픽 카드')
        add('korean_brand','지포스 '+number+(' 티아이' if suffix else ''))
        add('spoken_digits',''.join(DIGITS[c] for c in number)+(' 티아이' if suffix else '')+' 카드')
    elif name.startswith('RTX A'):
        code=name.split()[1];add('workstation_short',code+' 워크스테이션 카드')
        add('korean_workstation','엔비디아 '+code+' 그래픽 카드')
    elif 'Ada' in name:
        n=name.split()[1];add('ada_korean','RTX '+n+' 에이다');add('ada_short',n+' Ada 워크스테이션 GPU')
    elif 'PRO' in name:
        add('pro_korean','RTX 프로 6000 블랙웰');add('pro_short','PRO 6000 Blackwell 카드')
    else:
        base=name.split()[0]
        add('accelerator_short',base+' 가속기')
        spoken={'T4':'티포','L4':'엘포','A10':'에이텐','A40':'에이사십','L40S':'엘사십에스','V100':'브이백','A100':'에이백','H100':'에이치백','H200':'에이치이백','B100':'비백','B200':'비이백'}
        if base in spoken:add('spoken_accelerator',spoken[base]+' GPU')
    return variants


def korean_number(n):
    digits='영일이삼사오육칠팔구'
    def small(v):
        out=''
        for place,unit in [(1000,'천'),(100,'백'),(10,'십'),(1,'')]:
            k,v=divmod(v,place)
            if k:out+=('' if k==1 and place>1 else digits[k])+unit
        return out
    a,b=divmod(n,10000)
    return (small(a)+'만' if a else '')+small(b)

