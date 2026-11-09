// Minimal Markdown -> HTML for MODEL.md: headings, paragraphs, nested lists (list items may hold tables and
// further lists), pipe tables, horizontal rules, `code`, **bold**, *italic*, [links](url).

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function inline(s: string): string {
  const code: string[] = [];
  s = s.replace(/`([^`]+)`/g, (_, c) => `\u0000${code.push(esc(c)) - 1}\u0000`);
  s = esc(s)
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, u) => `<a href="${u}" target="_blank" rel="noopener">${t}</a>`)
    .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
    .replace(/\*([^*\s][^*]*)\*/g, '<i>$1</i>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${code[+i]}</code>`);
}

const LIST = /^([-*]|\d+\.)\s+/;
const indent = (l: string) => l.length - l.trimStart().length;
const cells = (l: string) => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());

function blocks(lines: string[]): string {
  let out = '', i = 0;
  while (i < lines.length) {
    const l = lines[i], t = l.trim();
    if (!t) { i++; continue; }
    let m: RegExpMatchArray | null;
    if ((m = t.match(/^(#{1,6})\s+(.*)$/))) { out += `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`; i++; continue; }
    if (/^-{3,}$/.test(t)) { out += '<hr>'; i++; continue; }
    if (t.startsWith('|') && i + 1 < lines.length && /^\|[\s|:-]+\|$/.test(lines[i + 1].trim())) {
      const head = cells(t); i += 2;
      let body = '';
      while (i < lines.length && lines[i].trim().startsWith('|')) { body += `<tr>${cells(lines[i]).map(c => `<td>${inline(c)}</td>`).join('')}</tr>`; i++; }
      out += `<table><thead><tr>${head.map(c => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table>`;
      continue;
    }
    if ((m = t.match(LIST))) {
      const ordered = /\d/.test(m[1]), base = indent(l);
      let items = '';
      while (i < lines.length) {
        const li = lines[i], mm = li.trim().match(LIST);
        if (!mm || indent(li) !== base || /\d/.test(mm[1]) !== ordered) break;
        const ci = base + mm[0].length;            // content indent of this item
        const body = [li.slice(ci)]; i++;
        while (i < lines.length && (!lines[i].trim() || indent(lines[i]) >= ci)) { body.push(lines[i].slice(ci)); i++; }
        while (body.length && !body[body.length - 1].trim()) body.pop();
        const html = blocks(body), single = html.match(/^<p>((?:(?!<p>)[\s\S])*)<\/p>$/);   // tight item: no <p>
        items += `<li>${single ? single[1] : html}</li>`;
      }
      out += ordered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`;
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|\||-{3,}$)/.test(lines[i].trim()) && !LIST.test(lines[i].trim())) para.push(lines[i++].trim());
    out += `<p>${inline(para.join(' '))}</p>`;
  }
  return out;
}

export function markdownToHtml(md: string): string { return blocks(md.replace(/\r\n/g, '\n').split('\n')); }
