//
//  ViewController.h
//  model-run
//
//  Created by Apple on 2021/7/9.
//

#import <UIKit/UIKit.h>

@interface ViewController : UIViewController

@property (weak, nonatomic) IBOutlet UISwitch *runGPU;
@property (weak, nonatomic) IBOutlet UITextView *resultsView;
@property (weak, nonatomic) IBOutlet UIButton *runPredict;

@end

