$ErrorActionPreference = 'Stop'
$src = 'C:\Users\panxy1019\Desktop\入党志愿书电子版(4)(1).doc'
$out = 'C:\Users\panxy1019\Desktop\入党志愿书_修订版.docx'
$pairs = @(
 @('庄严的提出','郑重地提出'),
 @('服务同学、奉献同学','服务同学、奉献集体'),
 @('主动倾听并尽力解决同学们学习、生活中遇到的困难','主动倾听同学们的诉求，尽力帮助他们解决学习和生活中遇到的困难'),
 @('如果党组织对我的申请表示不批准','如果党组织暂未批准我的申请'),
 @('缩短自己的差距','缩小自身与党员标准之间的差距'),
 @('改掉缺点和弥补不足','改正缺点、弥补不足'),
 @('持续向优秀老师和同学学习','虚心向老师和优秀同学学习'),
 @('邓小平理论和 “三个代表” 重要思想','邓小平理论、“三个代表”重要思想'),
 @('我深深感受到中国共产党的伟大与艰辛','我深深感受到中国共产党的伟大，以及党一路走来的艰辛与不易'),
 @('在革命、建设和改革各个历史时期，党始终坚持以人民为中心的发展思想','在革命、建设和改革各个历史时期，党始终坚持全心全意为人民服务的根本宗旨'),
 @('加入中国共产党是一项严肃的政治任务','加入中国共产党是一个郑重的政治选择'),
 @('保守党的机密','保守党的秘密'),
 @('坚决拥护党的决定','坚决执行党的决定'),
 @('争取早日在思想上入党，进而在组织上入党','不断提高思想觉悟，以实际行动争取早日加入党组织')
)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
 $doc = $word.Documents.Open($src, $false, $true)
 $original = $doc.Content.Text
 $expected = $original
 foreach ($pair in $pairs) {
  if (-not $expected.Contains($pair[0])) { throw "Missing text: $($pair[0])" }
  $expected = $expected.Replace($pair[0], $pair[1])
 }
 $doc.SaveAs2($out, 16)
 $word.UserName = '文字校对'
 $doc.TrackRevisions = $true
 foreach ($pair in $pairs) {
  $range = $doc.Content
  $range.Find.ClearFormatting()
  $range.Find.MatchWildcards = $false
  if (-not $range.Find.Execute($pair[0])) { throw "Find failed: $($pair[0])" }
  $range.Text = $pair[1]
 }
 $doc.ShowRevisions = $true
 $doc.Save()
 $count = $doc.Revisions.Count
 # Validate a disposable in-memory copy against the exact intended text.
 $verifyPath = Join-Path $env:TEMP 'party-review-verify.docx'
 $doc.SaveAs2($verifyPath, 16)
 $doc.AcceptAllRevisions()
 if ($doc.Content.Text -cne $expected) { throw 'Accepted text does not match expected exact replacements.' }
 $doc.Close(0)
 Write-Output "Verified $($pairs.Count) replacements; $count tracked revisions. Output: $out"
} finally { try { $word.Quit() } catch {} }
