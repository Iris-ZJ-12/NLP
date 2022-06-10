import json
import os
import re
import sys
import time
from glob import glob

parentDir = os.path.abspath(os.path.dirname(__file__))
# do not have O and U, while X represent any aa, B represent [D|N], Z represent [E|Q] and J represent [I|L]
standard_aa = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W",
               "X", "Y", "Z"]  # total 24 aa
aaList = {"Ala": "A", "Arg": "R", "Asn": "N",
          "Asp": "D", "Cys": "C", "Gln": "Q",
          "Glu": "E", "Gly": "G", "His": "H",
          "Ile": "I", "Leu": "L", "Lys": "K",
          "Met": "M", "Phe": "F", "Pro": "P",
          "Ser": "S", "Thr": "T", "Trp": "W",
          "Tyr": "Y", "Val": "V", "Asx": "B",
          "Xle": "J", "Glx": "Z", "Xaa": "X"}

PROTEIN = "protein"
NUCLEOTIDE = "nucleotide"
SOFTWARE_BIN = "/home/ryz/software/ncbi-blast-2.12.0+/bin/"
DNA = set("ACGTUN")
db_prt_total = "pat,nr"
db_prt_pat = "pat"
db_prt_nr = "nr"
db_nucl_total = "pat,nt"
db_nucl_pat = "pat"
db_nucl_nr = "nt"
# used for correct e-value when searching in splited database
dbsize = "XXXXX"
nthread = 8
# query number more than this value will divide all query into 2 file
divide_threshold = 50
# sequence length less than short_len should use another set of parameters
short_prt_len = 30  # use blastp-short
short_nucl_len = 50  # blastn-short


def correctPro(seq):
    seqNew = ""
    # remove digital ,space and symbol
    seq = re.sub("['\W','_','\d']", "", seq)
    if seq[:3] in aaList:
        if len(seq) % 3 != 0:
            print("May have mistake")
        for i in range(0, len(seq), 3):
            seqNew += aaList.get(seq[i:i + 3], "X")
    else:
        aas = "".join(aaList.values())
        # convert letter which is not aa to X
        seqNew = re.sub("[^%s]" % aas, "X", seq.upper())
    return seqNew


def correctDNA(seq):
    seq = re.sub("U", "T", seq, flags=re.IGNORECASE)
    seq = re.sub("['\W','_','\d']", "", seq)
    seq = re.sub("[^ATGC]", "N", seq.upper())
    return seq


def parse_input(sequence, seqtype, seqfile, message):
    # all input in the input frame will send as a long string
    fa_temp_file = open(seqfile, 'w')
    sequence = sequence.strip()
    if sequence.startswith(">"):  # multiple seqs
        # string like ">seq1 CTAGGGA >seq2 GCTAGTGG"
        seqs_tmp = sequence.split()
        ids = [seqs_tmp[i] for i in range(0, len(seqs_tmp), 2)]
        seqs = [seqs_tmp[i] for i in range(1, len(seqs_tmp), 2)]
        for id, seq in zip(ids, seqs):
            # check sequence type
            seqSet = set(seq)
            if seqSet.issubset(DNA) and seqtype != NUCLEOTIDE:
                pass

            # correct
            sequence = correctDNA(sequence) if seqtype == NUCLEOTIDE else correctPro(sequence)

            fa_temp_file.write(id + "\n")
            fa_temp_file.write(seq + "\n")


    else:  # regard as one seq
        sequence = correctDNA(sequence) if seqtype == NUCLEOTIDE else correctPro(sequence)
        fa_temp_file.write(">query" + "\n")
        fa_temp_file.write(sequence + "\n")
    fa_temp_file.close()
    return message


def creat_file(subdir):
    f_name = time.strftime("%Y_%m_%d_%H_%M_%S.fa", time.localtime())
    tmp_file = os.path.join(parentDir, "data_tmp", subdir, f_name)
    tmp_file_out = tmp_file + ".result.txt"
    return tmp_file, tmp_file_out

    # validSeq = []
    # isProtein = True
    # # remove space
    # seqlist = [i for i in seqlist if i != ""]

    # if not seqlist[0].startswith(">"):
    #     # set fasta identifier
    #     validSeq.append(">query")
    #     seq_temp = seqlist[0]
    # else:
    #     seq_temp = seqlist[1]
    # # judge whether seq is DNA or pro
    # seq_temp = re.sub("[^A-Z]]","",seq_temp.upper())
    # seq_temp_key = Counter(seq_temp).keys()
    # if set(seq_temp_key) <= {"A","T","C","G","U"}:
    #     isProtein = False
    # whole_seq = ""
    # for seq in seqlist:
    #     if seq.startswith(">"):
    #         fa_temp_file.wriet(seq+"\n")
    #     else:
    #         if isProtein:
    #             seq = correctPro(seq)
    #         else:
    #             seq = correctDNA(seq)
    #         fa_temp_file.wriet(seq+"\n")
    # fa_temp_file.close()
    # return isProtein,fa_temp


class Blast:
    def __init__(self):
        # database init
        self.prodb_pat = "/large_files/bio_seq/db/ncbi_embl_ddbj/pat_prt"
        self.prodb_nr = "/large_files/bio_seq/db/nr/nr"
        self.prodb_pat_ig = "/large_files/bio_seq/db/ncbi_embl_ddbj/antibody/picked_pat_ab"
        self.prodb_nr_ig = ""

        self.nucldb_pat = "/large_files/bio_seq/db/ncbi_embl_ddbj/pat_nucl"
        self.nucldb_nt = "/large_files/bio_seq/db/nt/nt"
        self.nuclDB = {
            "total": {db_nucl_total: '"' + self.nucldb_pat + ' ' + self.nucldb_nt + '"',
                      db_nucl_pat: self.nucldb_pat,
                      db_nucl_nr: self.nucldb_nt}
        }
        self.prtDB = {
            "total": {db_prt_total: '"' + self.prodb_pat + ' ' + self.prodb_nr + '"',
                      db_prt_pat: self.prodb_pat,
                      db_prt_nr: self.prodb_nr},
            "antibody": {db_prt_total: '"' + self.prodb_pat_ig + ' ' + self.prodb_nr_ig + '"',
                         db_prt_pat: self.prodb_pat_ig,
                         db_prt_nr: self.prodb_nr_ig}
        }

        # algorithm
        self.blastn = SOFTWARE_BIN + "blastn"
        self.blastp = SOFTWARE_BIN + "blastp"
        self.blastpm = SOFTWARE_BIN + "blastp"  # for motif should be special compiled
        self.blastx = SOFTWARE_BIN + "blastx"
        self.tblastn = SOFTWARE_BIN + "tblastn"

    def getDB(self, dbtype, dbname, message, scope="total"):
        '''parse parameters to get db '''
        dbs = self.prtDB if dbtype == PROTEIN else self.nuclDB
        try:
            return dbs[scope][dbname], message
        except:
            message += "#ERROR: unknown protein database name. "
        return "", message

    def getAlgorithm(self, seqtype, dbtype, message, task):
        if seqtype == PROTEIN and dbtype == PROTEIN:
            blast = self.blastp
        elif seqtype == NUCLEOTIDE and dbtype == NUCLEOTIDE:
            blast = self.blastn
        elif seqtype == NUCLEOTIDE and dbtype == PROTEIN:
            blast = self.blastx
        elif seqtype == PROTEIN and dbtype == NUCLEOTIDE:
            blast = self.tblastn
        else:
            message += "#ERROR: unknown blast type. "
            blast = ""
        prefix = os.path.basename(blast)
        #  task is subcommand of each blast
        if (not task.startswith(prefix)) and task != "megablast":
            message += "#ERROR: subcommand is not in accord with algorithm. "

        blast_alg = blast + " -task " + task
        return blast_alg, message

    def getQueryFile(self, query, seqtype, message):
        file_in, file_out = creat_file("sequence")
        message = parse_input(query, seqtype, file_in, message)
        return file_in, file_out, message

    ##################  sequence search ##########################################
    def sequence_search(self, query, seqtype, dbtype, dbname, task, evalue, wordsize, gapopen, gapextend, matrix,
                        penalty, reward, ungapped, max_target_num):
        # get parameters config
        message = ""
        db, message = self.getDB(dbtype, dbname, message)
        blast, message = self.getAlgorithm(seqtype, dbtype, message, task)
        # save input sequence to temp file and check
        tmp_file, tmp_file_out, message = self.getQueryFile(query, seqtype, message)
        # return the error message 
        if "ERROR" in message:
            # pass
            return "Database or task error"

        ungapp_str = "-ungapped -comp_based_stats F" if ungapped else ""
        mainparas = {
            "blast": blast, "query": tmp_file, "db": db, "out": tmp_file_out, "evalue": evalue,
            "wordsize": wordsize, "gapopen": gapopen, "gapextend": gapextend, "penalty": penalty,
            "reward": reward, "matrix": matrix, "ungapped": ungapp_str, "max_target_num": max_target_num,
            "num_threads": nthread}  # -outfmt  shoud redefine

        subcmd = " -query {query} -db {db} -out {out} -evalue {evalue} -word_size {wordsize} " \
                 "-gapopen {gapopen} -gapextend {gapextend} {ungapped} -outfmt 15 -num_threads {num_threads} " \
                 "-max_target_seqs {max_target_num}".format(**mainparas)

        supp_cmd = ""
        if task.startswith("blastn") or task.startswith("megablast"):
            supp_cmd = " -penalty {penalty} -reward {reward}  ".format(**mainparas)
            # cmd = "{self.blastn}  {subcmd} -penalty {penalty} -reward {reward} ".format(subcmd=subcmd,**mainparas,self=self)

        cmd = blast + " " + subcmd + " " + supp_cmd
        print(cmd)
        exitcode = os.system(cmd)
        if exitcode == 0:
            # result = "".join(open(tmp_file_out).readlines())
            contents = json.load(open(tmp_file_out))
            result = contents["BlastOutput2"]

        else:
            result = "run sequence_search error"
        return result

    def ab_seq_save(self, hcdr1, hcdr2, hcdr3, lcdr1, lcdr2, lcdr3, light, heavy):
        # set input and output filenames
        f_name = time.strftime("%Y_%m_%d_%H_%M_%S.fa", time.localtime())
        tmp_cdr = os.path.join(parentDir, "data_tmp", "antibody", f_name + ".cdr")
        tmp_cdr_out = tmp_cdr + ".result.txt"
        tmp_chain = os.path.join(parentDir, "data_tmp", "antibody", f_name + ".chain")
        tmp_chain_out = tmp_chain + ".result.txt"

        # save sequences into files
        haveCDR = False
        haveChain = False
        cdr_heads = ["hcdr1", "hcdr2", "hcdr3", "lcdr1", "lcdr2", "lcdr3"]
        with open(tmp_cdr, 'w') as fcdr:
            for i, seq in enumerate([hcdr1, hcdr2, hcdr3, lcdr1, lcdr2, lcdr3]):
                if seq:
                    seq = correctPro(seq)
                    haveCDR = True
                    fcdr.write(">%s\n" % cdr_heads[i])
                    fcdr.write(seq + "\n")

        hl_heads = ["light_chain", "heavy_chain"]
        with open(tmp_chain, 'w') as fchain:
            for i, seq in enumerate([light, heavy]):
                if seq:
                    seq = correctPro(seq)
                    haveChain = True
                    fchain.write(">%s\n" % hl_heads[i])
                    fchain.write(seq + "\n")
        return haveCDR, tmp_cdr, tmp_cdr_out, haveChain, tmp_chain, tmp_chain_out

    '''
    This function is used for counting CDR overlap
    '''

    def cdr_overlap(self, res):
        hits = {"hcdr1": set(), "hcdr2": set(), "hcdr3": set(), "lcdr1": set(), "lcdr2": set(), "lcdr3": set()}
        hits_olp = {}
        hits_olp_num = {}
        for cdr in res["BlastOutput2"]:
            tmp = cdr["report"]["results"]["search"]
            query_title = tmp["query_title"]
            for hit in tmp["hits"]:
                for one in hit["description"]:  # for seq have multiple id
                    hits[query_title].add(one["id"])

        hits_olp["hcdr123"] = hits["hcdr1"] & hits["hcdr2"] & hits["hcdr3"]
        hits_olp["hcdr12"] = hits["hcdr1"] & hits["hcdr2"]
        hits_olp["hcdr13"] = hits["hcdr1"] & hits["hcdr3"]
        hits_olp["hcdr23"] = hits["hcdr2"] & hits["hcdr3"]

        hits_olp["lcdr123"] = hits["lcdr1"] & hits["lcdr2"] & hits["lcdr3"]
        hits_olp["lcdr12"] = hits["lcdr1"] & hits["lcdr2"]
        hits_olp["lcdr13"] = hits["lcdr1"] & hits["lcdr3"]
        hits_olp["lcdr23"] = hits["lcdr2"] & hits["lcdr3"]

        for i in hits_olp:
            hits_olp_num[i] = len(hits_olp[i])

        return hits, hits_olp, hits_olp_num

    '''
    This function is used for CDR and antibody sequence blast
    '''

    def cdr_blast(self, dbtype, dbname, cdr_eval, hl_eval, cdr_wsize, hl_wsize, cdr_matrix, hl_matrix, \
                  cdr_gp_open, cdr_gp_extend, hl_gp_open, hl_gp_extend, hl_ungap, \
                  hcdr1, hcdr2, hcdr3, \
                  lcdr1, lcdr2, lcdr3, light, heavy, max_target_num):

        ##################  get run parameter and command #######################
        message = ""
        result_cdr = ""
        result_chain = ""
        searchDB, message = self.getDB(dbtype, dbname, message, scope="antibody")
        if "ERROR" in message:
            return "# ERROR: database name error!"

        haveCDR, cdr_seq, cdr_out, haveChain, hl_seq, hl_out = self.ab_seq_save(hcdr1, hcdr2, hcdr3, lcdr1, lcdr2,
                                                                                lcdr3, light, heavy)
        ungapp_str = "-ungapped -comp_based_stats F" if hl_ungap else ""
        supp_cmd = "-db {db} -max_target_seqs {num} -outfmt 15 -num_threads {nthread}".format(db=searchDB,
                                                                                              num=max_target_num,
                                                                                              nthread=nthread)

        ##################  blast CDR ##########################################
        exitcode_cdr, exitcode_chain = 0, 0
        if haveCDR:
            cmd = self.blastp + " -task blastp-short -query {filein} -evalue {eval}  -out {out} {supp_cmd} {gap} ".format(
                filein=cdr_seq, eval=cdr_eval, out=cdr_out, supp_cmd=supp_cmd, gap="-ungapped -comp_based_stats F")
            print(cmd)
            exitcode_cdr = os.system(cmd)
            # summary result
            if exitcode_cdr == 0:
                contents = json.load(open(cdr_out))
                result_cdr = contents["BlastOutput2"]
                # statistic CDR overlap
                hits, hits_olp, hits_olp_num = self.cdr_overlap(contents)

            else:
                result_cdr = "CDR aligment run error"

        ##################  handle light or heavy chain #######################
        if haveChain:
            cmd = self.blastp + " -task blastp -query {filein} -evalue {eval} -word_size {wz} -matrix {mx} \
            -gapopen {gopen} -gapextend {gext}  -out {out} {supp_cmd}  {ungapp}".format( \
                filein=hl_seq, eval=hl_eval, wz=hl_wsize, mx=hl_matrix, gopen=hl_gp_open, \
                gext=hl_gp_extend, out=hl_out, supp_cmd=supp_cmd, ungapp=ungapp_str)
            print(cmd)
            exitcode_chain = os.system(cmd)
            if exitcode_chain == 0:
                contents = json.load(open(hl_out))
                result_chain = contents["BlastOutput2"]
            else:
                result_chain = "light or heavy chain aligment run error"

        return {"CDR": result_cdr, "CDR_overlap": hits_olp, "CDR_overlap_num": hits_olp_num,
                "result_chain": result_chain}

    '''
    This function is used for parsing motif pattern into concrete sequences.
    '''

    def parse_pattern(self, pattern, seqList, querytype):
        # first parse multiple character [DFHY]
        aaset = re.search("\[([A-Za-z]+)\]?", pattern)
        if aaset:
            for i in set(aaset.group(1)):
                pattern_exp = re.sub("\[([A-Za-z]+)\]?", i, pattern)
                self.parse_pattern(pattern_exp, seqList, querytype)
        else:
            # second for DAE{3}AX{2,3}M, parse variable length {3} or {2,3}
            tmp = re.search("([A-Za-z])\{(\d,?[1-9]?)\}?", pattern)
            if tmp:
                tmp_len = list(map(int, tmp.group(2).split(",")))
                if len(tmp_len) == 1:  # {3}
                    char_exp = tmp.group(1) * tmp_len[0]
                    new_pattern = re.sub("[A-Za-z]\{\d,?[1-9]?\}?", char_exp, pattern, count=1)
                    self.parse_pattern(new_pattern, seqList, querytype)
                elif len(tmp_len) == 2:  # {2,3}
                    for i in range(tmp_len[0], tmp_len[1] + 1):
                        char_exp = tmp.group(1) * i
                        new_pattern = re.sub("[A-Za-z]\{\d,?[1-9]?\}?", char_exp, pattern, count=1)
                        self.parse_pattern(new_pattern, seqList, querytype)
                else:
                    sys.exit("### parse_pattern: illegal length style in motif")

            else:
                # third, replace N with A|T|G|C for nucleotide sequence
                if querytype != PROTEIN:
                    nuclN = re.search("N", pattern)
                    if nuclN:
                        for i in "ATCG":
                            new_pattern = re.sub("N", i, pattern, count=1)
                            self.parse_pattern(new_pattern, seqList, querytype)
                    else:
                        # save sequence after parse all pattern
                        seqList.append(pattern)
                else:
                    seqList.append(pattern)

                # save sequence after parse all pattern
                seqList.append(pattern)
            # pass reference because list is alterable

    '''
    This function is used for saving all concrete sequences coming from motif pattern.
    Long and short sequences would be saved in different files and use certain parameters in alignment.
    Querys will be divided into two files to allow for parallelization in the future.
    '''

    def motif_save(self, pattern, querytype):
        message = "ok"
        f_name = time.strftime("%Y_%m_%d_%H_%M_%S.fa", time.localtime())
        # get parameters for protein and nucleotide query
        qtype = "prt" if querytype == PROTEIN else "nucl"
        len_threshold = short_prt_len if querytype == PROTEIN else short_nucl_len

        valid_seq = []
        pattern = re.sub("[^\{\}\[\]\w\.,]", "", pattern)  # remove illegal character
        pattern = pattern.replace("_", "")  # remove illegal character
        if querytype == PROTEIN:
            pattern = pattern.replace(".", "X")  # replace dot with X
        else:
            pattern = pattern.replace(".", "N")
        # resolver all combination from the pattern
        self.parse_pattern(pattern, valid_seq, querytype)

        seq_num = len(valid_seq)
        if seq_num > 100:
            # report error and do not run blast
            message = "### motif_save: pattern exceed 100 combination"
            return message, ""

        seqs = {}
        seqs["short"] = [i for i in valid_seq if len(i) <= len_threshold]
        seqs["long"] = [i for i in valid_seq if len(i) > len_threshold]
        # for short and long sequence 
        for length in seqs:
            # the default file to save querys
            filename = ".".join([f_name, length, qtype, "fa1"])
            tmp_file1 = os.path.join(parentDir, "data_tmp", "motif", filename)
            mf_fh1 = open(tmp_file1, 'w')

            item_num = len(seqs[length])
            # should save into 2 files with so much querys
            if item_num > divide_threshold:
                filename = filename = ".".join([f_name, length, qtype, "fa2"])
                tmp_file2 = os.path.join(parentDir, "data_tmp", "motif", filename)
                mf_fh2 = open(tmp_file2, "w")

                middl = item_num // 2
                for i, one in enumerate(seqs[length]):
                    if i < middl:
                        mf_fh1.write(">" + str(i) + "\n")
                        mf_fh1.write(one + "\n")
                    else:
                        mf_fh2.write(">" + str(i) + "\n")
                        mf_fh2.write(one + "\n")
                mf_fh2.close()
            else:
                for i, one in enumerate(seqs[length]):
                    mf_fh1.write(">" + str(i) + "\n")
                    mf_fh1.write(one + "\n")
            mf_fh1.close()
        return message, f_name, seq_num

    '''
    Do motif blast
    '''

    def motif_blast(self, pattern, seqtype, dbtype, dbname, evalue):  # pattern,seqtype,dbtype,dbname,evalue
        message = ""
        db, message = self.getDB(dbtype, dbname, message)
        message, f_name, seq_num = self.motif_save(pattern, seqtype)

        if message != "ok":
            return {"message": message}

        # get parameters for each type search
        qtype = "prt" if seqtype == PROTEIN else "nucl"
        if seqtype == PROTEIN:
            blast = self.blastpm
            blast_short = " -task blastp-short "
            ungap = " -ungapped -comp_based_stats F "
        else:
            blast = self.blastn
            blast_short = " -task blastn-short "
            ungap = " -ungapped "

        # get query files
        sufx_s = ".".join([f_name, "short", qtype, "fa*"])
        sufx_l = ".".join([f_name, "long", qtype, "fa*"])
        m_s_files = glob(os.path.join(parentDir, "data_tmp", "motif", sufx_s))
        m_l_files = glob(os.path.join(parentDir, "data_tmp", "motif", sufx_l))

        sub_cmd = "  -db {db} -evalue {eval} -outfmt 15 -num_threads {nm} ".format(db=db, eval=evalue, nm=nthread)

        if len(m_s_files) > 0:
            for shortseq in m_s_files:
                shortseq_out = shortseq + ".result.txt"
                cmd = blast + blast_short + " -query {filein}  -out {out}".format(filein=shortseq,
                                                                                  out=shortseq_out) + sub_cmd + ungap
                print(cmd)
                exitcode = os.system(cmd)
                if exitcode != 0:
                    return {"message": "blast error for short sequence"}

        if len(m_l_files) > 0:
            for longseq in m_s_files:
                longseq_out = longseq + ".result.txt"
                cmd = blast + blast_short + " -query {filein} -out {out}".format(filein=longseq,
                                                                                 out=longseq_out) + sub_cmd
                print(cmd)
                exitcode = os.system(cmd)
                if exitcode != 0:
                    return {"message": "blast error for long sequence"}

        # read all blast result
        results = []
        res_files = glob(os.path.join(parentDir, "data_tmp", "motif", f_name + "*.result.txt"))
        for i in res_files:
            # the value of BlastOutput2 is list containing all subjects corresponding each query
            contents = json.load(open(i))["BlastOutput2"]
            results.extend(contents)
            # results.extend(open(i).readlines())

        return {"message": message, "seqs_num": seq_num, "result": results}
