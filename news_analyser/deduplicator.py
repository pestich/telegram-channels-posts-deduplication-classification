from model import Model
from xxhash import xxh64_intdigest
from datasketch import MinHash, MinHashLSH
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class Deduplicator:
    def __init__(self):
        self.lsh_threshold = 0.6
        self.num_perms = 128
        self.shingle_size = 3
        self.sbert_threshold = 0.9
        self.model = Model.load_deduplicator()

    def shingle(self, string, shingle_size):
        string = string[:500]
        shings = {string[i: i + shingle_size] for i in range(len(string) - shingle_size + 1)}
        return set(shings)

    def drop_near_duplicates(self, data):
        min_dict = dict()

        for idx, text in tqdm(data.items()):
            shingles = self.shingle(str(text), self.shingle_size)
            mhash = MinHash(num_perm=self.num_perms, hashfunc=xxh64_intdigest)
            for shing in shingles:
                mhash.update(shing.encode("utf8"))
            min_dict[idx] = mhash

        lsh_high = MinHashLSH(threshold=0.8, num_perm=self.num_perms)
        for key in tqdm(min_dict.keys()):
            lsh_high.insert(key, min_dict[key])

        lsh_low = MinHashLSH(threshold=0.55, num_perm=self.num_perms)
        for key in tqdm(min_dict.keys()):
            lsh_low.insert(key, min_dict[key])

        cand_list_high = []
        for query in min_dict.keys():
            bucket = lsh_high.query(min_dict[query])
            if len(bucket) > 1:
                first_val = bucket[0]
                for val in bucket[1:]:
                    second_val = val
                    if [first_val, second_val] not in cand_list_high:
                        cand_list_high.append([first_val, second_val])

        cand_list_low = []
        for query in min_dict.keys():
            bucket = lsh_low.query(min_dict[query])
            if len(bucket) > 1:
                first_val = bucket[0]
                for val in bucket[1:]:
                    second_val = val
                    if ([first_val, second_val] not in cand_list_low) and (
                            [first_val, second_val]) not in cand_list_high:
                        cand_list_low.append([first_val, second_val])

        drop_list = []
        for i_candidate in cand_list_low:
            sent_vec_0 = self.model.encode(data[i_candidate[0]])
            sent_vec_1 = self.model.encode(data[i_candidate[1]])
            sent_score = cosine_similarity([sent_vec_0], [sent_vec_1]).tolist()[0][0]
            if sent_score > self.sbert_threshold:
                if len(data[i_candidate[0]]) <= len(data[i_candidate[1]]):
                    drop_list.append(i_candidate[0])
                else:
                    drop_list.append(i_candidate[1])

        for i_candidate in cand_list_high:
            if len(data[i_candidate[0]]) <= len(data[i_candidate[1]]):
                drop_list.append(i_candidate[0])
            else:
                drop_list.append(i_candidate[1])

        drop_list = list(set(drop_list))

        return drop_list
