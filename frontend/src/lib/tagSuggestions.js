// 태그 자동완성용 후보 목록 (Danbooru식).
// PromptHelper의 그룹 태그 + 자주 쓰는 태그를 합친 것.
export const TAG_SUGGESTIONS = [
  // 인물
  '1girl', '1boy', '2girls', 'solo', 'loli', 'shota',
  // 얼굴/눈
  'detailed eyes', 'beautiful detailed eyes', 'looking at viewer',
  'looking away', 'closed eyes', 'smile', 'blush', 'open mouth',
  // 머리
  'long hair', 'short hair', 'twintails', 'ponytail', 'hime cut',
  'black hair', 'brown hair', 'blonde hair', 'silver hair', 'blue hair',
  'pink hair', 'red hair', 'ahoge',
  // 몸/포즈
  'upper body', 'full body', 'cowboy shot', 'portrait', 'close-up',
  'dynamic pose', 'sitting', 'standing', 'lying',
  // 의상
  'school uniform', 'serafuku', 'dress', 'maid', 'swimsuit', 'kimono',
  'hoodie', 'sailor collar', 'pleated skirt', 'thighhighs',
  // 품질
  'masterpiece', 'best quality', 'highres', 'absurdres', 'newest',
  'highly detailed', 'score_9', 'score_8_up',
  // 화풍
  'anime style', 'anime screencap', 'cel shaded', 'soft shading',
  'watercolor', 'vibrant colors', 'soft lighting', 'cinematic lighting',
  // 배경
  'simple background', 'white background', 'outdoors', 'indoors',
  'sky', 'sunset', 'night', 'cherry blossoms', 'beach', 'classroom',
  // 기타
  'soft skin', 'detailed face', 'clean outlines',
];

/** 입력 중 draft에 대한 후보 (최대 8개, 대소문자 무시 부분일치) */
export function suggestTags(draft, exclude = []) {
  const q = draft.trim().toLowerCase();
  if (!q) return [];
  const excluded = new Set(exclude.map((t) => t.toLowerCase()));
  return TAG_SUGGESTIONS.filter(
    (t) => t.toLowerCase().includes(q) && !excluded.has(t.toLowerCase())
  ).slice(0, 8);
}
