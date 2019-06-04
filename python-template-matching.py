import cv2
import numpy as np

def template_matching(src, temp):
    # .shapeで画像の高さ・幅を取得
    h = src.shape[0]
    w = src.shape[1]

    ht = temp.shape[0]
    wt = temp.shape[1]

    # スコア格納用の二次元配列
    # ソース画像 - テンプレート画像 が行・列の数になる
    score = np.empty((h - ht, w - wt))

    # for文で走査していく
    for dy in range(0, h - ht):
        for dx in range(0, w - wt):
            # 差分の絶対和を計算
            diff = np.abs(src[dy:dy + ht, dx:dx + wt] - temp)
            score[dy, dx] = diff.sum()

    # スコアが最小の走査位置を返す
    pt = np.unravel_index(score.argmin(), score.shape)

    return(pt[1], pt[0])

def main():
    # 入力画像とテンプレート画像をimreadで取得
    img = cv2.imread('imput.png')
    temp = cv2.imread('temp.png')

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    # テンプレート画像の高さ・幅
    h, w = temp.shape

    # テンプレートマッチング(NumPyで実装)
    pt = template_matching(gray, temp)

    # テンプレートマッチングの結果を出力
    cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
    cv2.imwrite('output.png', img)

main()