"""Read-only parser/checker for this bibliography; BibTeX is the syntax authority.

This supports the braced/quoted/macro fields used by the acquired exports.
It deliberately does not import RIS, NBIB, CSL, or plain-text citations.
"""
import re
from collections import Counter
from pathlib import Path


def entries(text):
    result = []
    pattern = re.compile(r'@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,')
    pos = 0
    while match := pattern.search(text, pos):
        fields = {}
        i = match.end()
        while True:
            while text[i].isspace() or text[i] == ',':
                i += 1
            if text[i] == '}':
                i += 1
                break
            field = re.match(r'([A-Za-z][A-Za-z0-9_-]*)\s*=\s*', text[i:])
            if not field:
                raise ValueError(f'Invalid field near {text[i:i+60]!r}')
            name = field.group(1).lower()
            if name in fields:
                raise ValueError(f'Duplicate field {name} in {match.group(2)}')
            i += field.end()
            start = i
            if text[i] == '{':
                depth = 1
                i += 1
                while depth:
                    if text[i] == '\\':
                        i += 2
                        continue
                    depth += (text[i] == '{') - (text[i] == '}')
                    i += 1
            elif text[i] == '"':
                i += 1
                depth = 0
                while text[i] != '"' or depth:
                    if text[i] == '\\':
                        i += 2
                        continue
                    depth += (text[i] == '{') - (text[i] == '}')
                    i += 1
                i += 1
            else:
                while text[i] not in ',}':
                    i += 1
            fields[name] = text[start:i].strip()
        result.append({'type': match.group(1), 'key': match.group(2), 'fields': fields})
        pos = i
    return result


def value(raw):
    if raw.startswith('{') and raw.endswith('}'):
        return raw[1:-1]
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    return raw


def fully_grouped(s):
    if not s.startswith('{') or not s.endswith('}'):
        return False
    depth = 0
    i = 0
    while i < len(s):
        if s[i] == '\\':
            i += 2
            continue
        depth += (s[i] == '{') - (s[i] == '}')
        if depth == 0 and i != len(s)-1:
            return False
        i += 1
    return depth == 0


def serialize(entry):
    return ('@' + entry['type'] + '{' + entry['key'] + ',\n' +
            ''.join(f'  {k} = {v},\n' for k, v in entry['fields'].items()) + '}\n')


def citation_keys(tex):
    tex = re.sub(r'(?m)(?<!\\)%.*$', '', tex)
    return list(dict.fromkeys(k.strip() for c in re.findall(
        r'\\cite\w*\s*(?:\[[^\]]*\])?\{([^}]+)\}', tex)
        for k in c.split(',')))


def audit(folder):
    combined = (folder/'references_official.bib').read_text()
    data = entries(combined)
    keys = [e['key'] for e in data]
    errors = []
    duplicates = [k for k, n in Counter(keys).items() if n > 1]
    for e in data:
        f = e['fields']
        if 'title' in f:
            title = f['title']
            if not (fully_grouped(title) and fully_grouped(title[1:-1])):
                errors.append(f"{e['key']}: title is not fully double-braced")
            elif fully_grouped(title[2:-2]):
                errors.append(f"{e['key']}: redundant third whole-title brace layer")
        if 'month' in f and f['month'] not in 'jan feb mar apr may jun jul aug sep oct nov dec'.split():
            errors.append(f"{e['key']}: month is not a standard unquoted macro")
        for field, raw in f.items():
            if field not in ('url', 'doi') and re.search(r'(?<!\\)&', raw):
                errors.append(f"{e['key']}: unescaped ampersand in {field}")
        is_arxiv = value(f.get('archiveprefix', '')).lower() == 'arxiv'
        if is_arxiv and 'eprint' in f and 'arXiv:'+value(f['eprint']) not in value(f.get('note','')):
            errors.append(f"{e['key']}: arXiv identifier not visible in note")
    tex = (folder.parent/'main.tex').read_text()
    cited = citation_keys(tex)
    ctl = 'IEEEexample:BSTcontrol'
    expected = {'ctluse_forced_etal':'yes', 'ctlmax_names_forced_etal':'6',
                'ctlnames_show_etal':'1', 'ctluse_url':'no',
                'ctldash_repeated_names':'no'}
    controls = [e for e in data if e['key'] == ctl]
    control_valid = len(controls)==1 and all(value(controls[0]['fields'].get(k,''))==v for k,v in expected.items())
    return {'reference_count':sum(e['type'].lower()!='ieeetranbstctl' for e in data),
            'control_entry_count':len(controls), 'total_entries':len(data),
            'duplicate_keys':duplicates, 'format_errors':errors,
            'manuscript_cited_count':len(cited),
            'missing_citation_keys':[k for k in cited if k not in keys],
            'unexpected_reference_keys':[k for k in keys if k not in cited and k!=ctl],
            'control_valid':control_valid,
            'control_before_first_citation':tex.find('\\bstctlcite{'+ctl+'}') < tex.find('\\cite{'),
            'style_directive_correct':'\\bibliographystyle{IEEEtran}' in tex,
            'database_directive_correct':'\\bibliography{references_official}' in tex,
            'url_package_present':bool(re.search(r'\\usepackage(?:\[[^\]]*\])?\{[^}]*\b(?:url|hyperref)\b[^}]*\}',tex))}


if __name__ == '__main__':
    import json
    print(json.dumps(audit(Path(__file__).resolve().parent), indent=2))
