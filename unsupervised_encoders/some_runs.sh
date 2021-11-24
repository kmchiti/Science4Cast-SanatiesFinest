modes=('aa_ja_cn_pa_ra_pr'  'ja_cn_pa_ra_pr'  'aa_cn_pa_ra_pr'  'aa_ja_pa_ra_pr'  'aa_ja_cn_ra_pr'  'aa_ja_cn_pa_pr'  'aa_ja_cn_pa_ra'  'cn_pa_ra_pr'  'ja_pa_ra_pr'  'ja_cn_ra_pr'  'ja_cn_pa_pr'  'ja_cn_pa_ra'  'aa_pa_ra_pr'  'aa_cn_ra_pr'  'aa_cn_pa_pr'  'aa_cn_pa_ra'  'aa_ja_ra_pr'  'aa_ja_pa_pr'  'aa_ja_pa_ra'  'aa_ja_cn_pr'  'aa_ja_cn_ra'  'aa_ja_cn_pa'  'pa_ra_pr'  'cn_ra_pr'  'cn_pa_pr'  'cn_pa_ra'  'ja_ra_pr'  'ja_pa_pr'  'ja_pa_ra'  'ja_cn_pr'  'ja_cn_ra'  'ja_cn_pa'  'aa_ra_pr'  'aa_pa_pr'  'aa_pa_ra'  'aa_cn_pr'  'aa_cn_ra'  'aa_cn_pa'  'aa_ja_pr'  'aa_ja_ra'  'aa_ja_pa'  'aa_ja_cn'  'ra_pr'  'pa_pr'  'pa_ra'  'cn_pr'  'cn_ra'  'cn_pa'  'ja_pr'  'ja_ra'  'ja_pa'  'ja_cn'  'aa_pr'  'aa_ra'  'aa_pa'  'aa_cn'  'aa_ja'  'pr'  'ra'  'pa'  'cn'  'ja'  'aa')


for mode in "${modes[@]}"; do
  cmd="python3 train.py --features  $mode"
  echo $cmd
  $cmd
done