labels = ['A++轮', 'A+轮', 'A2轮', 'A轮', 'B+轮', 'B1轮', 'B轮',
                  'C+轮', 'C轮', 'D+轮', 'D1轮', 'D轮', 'E/E+轮', 'E+轮',
                  'E轮', 'F轮', 'G轮', 'H轮', 'IPO', 'I轮', 'Pre-A+轮',
                  'Pre-A轮', 'Pre-B+轮', 'Pre-B2轮', 'Pre-B轮', 'Pre-C轮',
                  'Pre-IPO', '并购', '其他', '天使轮', '增发', '战略融资',
                  '种子轮', 'R轮']

def refine_label(label):
    label = label.replace(' ', '').upper().replace('PRE-', 'Pre-')
    tmps = []
    for l in labels:
        if l in label:
            tmps.append(l)
    print(tmps)
    if tmps:
        label = max(tmps)
    return label


if __name__ == '__main__':
    l = 'B11111111'
    r = refine_label(l)
    print(r)
